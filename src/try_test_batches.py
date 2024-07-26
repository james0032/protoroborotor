import argparse
import os.path as osp

#torch and torch_geometric are already in the container where this is to run
import torch
import torch.optim as optim

#from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

from src.ROBOKOP_Data import ROBOKOP

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

from datetime import datetime as dt
@torch.no_grad()
def test(data, bs=20000, k=10):
    model.eval()
    print("Testing Model")
    start = dt.now()
    return model.test(
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
for trials in ( (100, 10), (1000, 10), (10000,10) ):
    test(val_data, bs=trials[0], k=trials[1])
