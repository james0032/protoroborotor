import argparse
import os.path as osp

#torch and torch_geometric are already in the container where this is to run
import torch
from torch import Tensor

#from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

from src.ROBOKOP_Data import ROBOKOP

from typing import Tuple
from tqdm import tqdm

from collections import defaultdict

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, required=True)
argparser.add_argument('--modelepoch', type=int, default=100)
argparser.add_argument('--model', type=str, default="rotate")
args = argparser.parse_args()

device = 'cuda'
print("CUDA?", torch.cuda.is_available())
path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  '..', 'robokop', args.dataset)

print(osp.realpath(__file__))

print("load data")
train_data = ROBOKOP(path, split='train')[0].to(device)
val_data = ROBOKOP(path, split='val')[0].to(device)
test_data = ROBOKOP(path, split='test')[0].to(device)

print("Number of training nodes:", train_data.num_nodes)
print("Number of relations:", train_data.num_edge_types)
model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)


## This is taken from pytorch geometric's  framework but modified to provide per-predicate results
def localtest(
        model,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = False,
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
    mean_ranks_by_pred = defaultdict(list)
    reciprocal_ranks_by_pred = defaultdict(list)
    hits_by_pred = defaultdict(list)
    counts = defaultdict(int)
    print("here we go...")
    for i in arange:
        h, r, t = head_index[i], rel_type[i], tail_index[i]

        scores = []
        tail_indices = torch.arange(model.num_nodes, device=t.device)
        for ts in tail_indices.split(batch_size):
            scores.append(model(h.expand_as(ts), r.expand_as(ts), ts))
        rank = int((torch.cat(scores).argsort(
            descending=True) == t).nonzero().view(-1))
        mean_ranks.append(rank)
        reciprocal_ranks.append(1 / (rank + 1))
        hits_at_k.append(rank < k)
        etype = r.item()
        mean_ranks_by_pred[etype].append(rank)
        reciprocal_ranks_by_pred[etype].append(1 / (rank + 1))
        hits_by_pred[etype].append(rank < k)
        counts[etype] += 1

    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
    mr_p = {}
    mrr_p = {}
    hits_p = {}
    for etype in mean_ranks_by_pred:
        mr_p[etype] = float(
            torch.tensor(mean_ranks_by_pred[etype], dtype=torch.float).mean()
        )
        mrr_p[etype] = float(
            torch.tensor(reciprocal_ranks_by_pred[etype], dtype=torch.float).mean()
        )
        hits_p[etype] = int(
            torch.tensor(hits_by_pred[etype]).sum()
        ) / len(hits_by_pred[etype])

    return mean_rank, mrr, hits_at_k, mr_p, mrr_p, hits_p, counts

@torch.no_grad()
def test(data, bs=20000, k=10):
    model.eval()
    print("Testing Model")
    return localtest(
        model,
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=bs,
        k=k,
        log=False
    )

stuff = torch.load(f"{path}/model_{args.modelepoch}.pt")
model.load_state_dict(stuff['model_state_dict'])

# Now lets see how long it takes to run test?
batchsize = 20000
k = 10
with open(f"{path}/results_{args.modelepoch}.txt", 'w') as f:
    f.write("Predicate\tCounts\tMean Rank\tMRR\tHits@10\n")
    mean_rank, mrr, hits_at_k, mr_p, mrr_p, hits_p, counts = test(test_data, bs=batchsize, k=k)
    print(f"Mean Rank: {mean_rank}, MRR: {mrr}, Hits@10: {hits_at_k}")
    f.write(f"All\t{sum(counts.values())}\t{mean_rank}\t{mrr}\t{hits_at_k}\n")
    for e in mr_p:
        print(f"Predicate: {e} Counts: {counts[e]}  Test Mean Rank: {mr_p[e]:.2f}, Test MRR: {mrr_p[e]:.4f}, Test Hits@10: {hits_p[e]:.4f}")
        f.write(f"{e}\t{counts[e]}\t{mr_p[e]:.2f}\t{mrr_p[e]:.4f}\t{hits_p[e]:.4f}\n")
