import argparse
import os.path as osp

#torch and torch_geometric are already in the container where this is to run
import torch
import torch.optim as optim
from torch import Tensor

#from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

from src.ROBOKOP_Data import ROBOKOP

from typing import Tuple
from tqdm import tqdm

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}


args={"model":"rotate", "dataset" : "CGD"}

device = 'cuda'
print("CUDA?", torch.cuda.is_available())
path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  '..', 'robokop', args['dataset'])

print(osp.realpath(__file__))

print("load data")
train_data = ROBOKOP(path, split='train')[0].to(device)
val_data = ROBOKOP(path, split='val')[0].to(device)
test_data = ROBOKOP(path, split='test')[0].to(device)

print("Number of training nodes:", train_data.num_nodes)
print("Number of relations:", train_data.num_edge_types)
model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args['model']](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args['model'], {}),
).to(device)


def localtest(
        model,
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
    print("HI")
    arange = range(head_index.numel())
    arange = tqdm(arange) if log else arange

    mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
    print("here we go...")
    for i in arange:
        print(i)
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

    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

    return mean_rank, mrr, hits_at_k

from datetime import datetime as dt
@torch.no_grad()
def test(data, bs=20000, k=10):
    model.eval()
    print("Testing Model")
    start = dt.now()
    return localtest(
        model,
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=bs,
        k=k,
        log=True
    )
    end = dt.now()
    print(f"batchsize={bs} k={k} took {end-start}")

stuff = torch.load(f"{path}/model_100.pt")
model.load_state_dict(stuff['model_state_dict'])

# Now lets see how long it takes to run test?
for trials in ( (10000, 10),):
    test(val_data, bs=trials[0], k=trials[1])
