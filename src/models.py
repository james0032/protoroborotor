import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200, input_drop=0.2, hidden_drop=0.3, feature_map_drop=0.2):
        super(ConvE, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb_shape1 = 10  # must satisfy emb_shape1 * emb_shape2 = embedding_dim
        self.emb_shape2 = embedding_dim // self.emb_shape1
        self.num_nodes = num_entities
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        self.inp_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_map_drop = nn.Dropout2d(feature_map_drop)

        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        self.fc = nn.Linear(32 * 2 * self.emb_shape1 * self.emb_shape2, embedding_dim)

        self.loss = nn.BCEWithLogitsLoss() #nn.BCELoss()
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_entities)))

    def forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor:
        # Compute scores for the given triplets
        e_h = self.entity_embedding(head_index)
        r = self.relation_embedding(rel_type)
        e_t = self.entity_embedding(tail_index)

        # Prepare input like in your original forward
        e = torch.cat([e_h, r], 1).view(-1, 1, 2 * self.emb_shape1, self.emb_shape2)

        x = self.bn0(e)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Compute score using dot product with tail embedding
        score = torch.sum(x * e_t, dim=1)
        return score

    def predict_all_tails(self, head, relation):
        e_h = self.entity_embedding(head)
        r = self.relation_embedding(relation)
        if e_h.dim() == 1:
            e_h = e_h.unsqueeze(0)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        # Reshape and concatenate
        e = torch.cat([e_h, r], 1).view(-1, 1, 2 * self.emb_shape1, self.emb_shape2)

        x = self.bn0(e)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # score against all entities
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0)) # [B, num_entities]
        x += self.bias
        pred = torch.sigmoid(x)
        return pred
    
    def loader(self, head_index, rel_type, tail_index, batch_size=1024, shuffle=True):
        #"""Mimics torch_geometric.nn.RotatE.loader behavior."""
        #triples = torch.stack([head_index, rel_type, tail_index], dim=1)
        #dataset = TensorDataset(triples)
        #return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataset = TripleDataset(head_index, rel_type, tail_index)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    
    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)
    
    @torch.no_grad()
    def test(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float, float]:
        r"""Evaluates the model quality by computing Mean Rank, MRR and
        Hits@:math:`k` across all possible tail entities.

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
            batch_size (int): The batch size to use for evaluating.
            k (int, optional): The :math:`k` in Hits @ :math:`k`.
                (default: :obj:`10`)
            log (bool, optional): If set to :obj:`False`, will not print a
                progress bar to the console. (default: :obj:`True`)
        """
        arange = range(head_index.numel())
        arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            
            h, r, t = head_index[i], rel_type[i], tail_index[i]
            #scores = []
            #tail_indices = torch.arange(self.num_nodes, device=t.device)
            #for ts in tail_indices.split(batch_size):
            #    scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            scores = self.predict_all_tails(h, r)
            scores = scores.view(-1)
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == t).nonzero(as_tuple=False).item()
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k

    @torch.no_grad()
    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index

    
class TripleDataset(Dataset):
    def __init__(self, head_index, rel_type, tail_index):
        assert len(head_index) == len(rel_type) == len(tail_index)
        self.head = head_index
        self.rel = rel_type
        self.tail = tail_index

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return self.head[idx], self.rel[idx], self.tail[idx]
