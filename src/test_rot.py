import argparse
import os.path as osp

#torch and torch_geometric are already in the container where this is to run
import torch
import torch.optim as optim

#from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

from src.ROBOKOP_Data import ROBOKOP
from src.Test_data import PredicateTestData

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}

# read an argument from the command line, specifying the name of the dataset
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, required=True)
argparser.add_argument('--modelepoch', type=int, default=100)
argparser.add_argument('--model', type=str, default="rotate")
args = argparser.parse_args()

device = 'cuda'
print("CUDA?", torch.cuda.is_available())
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#v1 is a full robokop baseline - subclass, with 80/20/20 split
path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  '..', 'robokop', args.dataset)

train_data = ROBOKOP(path, split='train')[0].to(device)

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get(args.model, {}),
).to(device)

pcount = train_data.num_edge_types

stuff = torch.load(f"{path}/model_{args.modelepoch}.pt")
model.load_state_dict(stuff['model_state_dict'])

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


with open(f"{path}/results_{args.modelepoch}.txt", 'w') as f:
    f.write("Predicate\tMean Rank\tMRR\tHits@10\n")
    for s in range(pcount):
        try:
            test_data = PredicateTestData(path, split=s, num_preds=pcount)[0].to(device)
            rank, mrr, hits_at_10 = test(test_data)
            print(f'Predicate: {s}   Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, ' f'Test Hits@10: {hits_at_10:.4f}')
            f.write(f'{s}\t{rank:.2f}\t{mrr:.4f}\t{hits_at_10:.4f}\n')
        except:
            print(f"No test edges for predicate:{s}")