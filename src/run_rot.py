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

# read an argument from the command line, specifying the name of the dataset
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, required=True)
argparser.add_argument('--model', choices=model_map.keys(), type=str.lower, default='rotate')
argparser.add_argument('--epochs', type=int, default=1000)
argparser.add_argument('--testrate', type=int, default=100)
argparser.add_argument('--saverate', type=int, default=100)
args = argparser.parse_args()

device = 'cuda'
print("CUDA?", torch.cuda.is_available())
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#v1 is a full robokop baseline - subclass, with 80/20/20 split
path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  '..', 'robokop', args.dataset)

print(osp.realpath(__file__))
print(path)

#I don't understand why these [0] are here
# just looking at the types, without the 0 is a ROBOKOP but with the 0 it's a Data
# So it looks like the whole thing is in memory if this is to believed
train_data = ROBOKOP(path, split='train')[0].to(device)
val_data = ROBOKOP(path, split='val')[0].to(device)
test_data = ROBOKOP(path, split='test')[0].to(device)

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)
print("Start data loader")
loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
    num_workers=8,
)
print("model loader done.")
optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map[args.model]


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    print("Testing Model")
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
        log=False
    )


for epoch in range(1, args.epochs):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % args.saverate == 0:
        print("Saving Model")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f"{path}/model_{epoch}.pt")
    if epoch % args.testrate == 0:
        rank, mrr, hits = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

print("One last test")
rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')
