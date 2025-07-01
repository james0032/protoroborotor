import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset, TensorDataset

class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200, input_drop=0.2, hidden_drop=0.3, feature_map_drop=0.2):
        super(ConvE, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb_shape1 = 10  # must satisfy emb_shape1 * emb_shape2 = embedding_dim
        self.emb_shape2 = embedding_dim // self.emb_shape1

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

        self.loss = nn.BCELoss()
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_entities)))

    def forward(self, head, relation):
        e_h = self.entity_embedding(head)
        r = self.relation_embedding(relation)

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
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        x += self.bias
        pred = torch.sigmoid(x)
        return pred
    
    def loader(self, head_index, rel_type, tail_index, batch_size=1024, shuffle=True):
        #"""Mimics torch_geometric.nn.RotatE.loader behavior."""
        #triples = torch.stack([head_index, rel_type, tail_index], dim=1)
        #dataset = TensorDataset(triples)
        #return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataset = TripleDataset(head_index, rel_type, tail_index)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    
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
        #arange = tqdm(arange) if log else arange

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            tail_indices = torch.arange(self.num_nodes, device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(self(h.expand_as(ts), r.expand_as(ts), ts))
            rank = int((torch.cat(scores).argsort(
                descending=True) == t).nonzero().view(-1))
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank + 1))
            hits_at_k.append(rank < k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k
    
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
