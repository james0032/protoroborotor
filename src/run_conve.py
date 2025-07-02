import argparse
import os.path as osp

#torch and torch_geometric are already in the container where this is to run
import torch
import torch.optim as optim
import torch.multiprocessing as mp
#from torch_geometric.datasets import FB15k_237
from torch_geometric.data import Data
from typing import List, Tuple
from tqdm import tqdm
from src.ROBOKOP_Data import ROBOKOP
from src.models import ConvE

def main(args):
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
    train_triples = data_to_triples(train_data)
    valid_triples = data_to_triples(val_data)
    test_triples = data_to_triples(test_data)   
    all_triples = generate_all_triples(train_triples, valid_triples, test_triples)
    #model_arg_map = {'rotate': {'margin': 9.0}}
    model = ConvE(
        num_entities=train_data.num_nodes,
        num_relations=train_data.num_edge_types,
        #embedding_dim=200, input_drop=0.2, hidden_drop=0.3, feature_map_drop=0.2
        #hidden_channels=50,
        #**model_arg_map.get(args.model, {}),
    ).to(device)
    print("Start data loader")
    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=train_data.edge_type,
        tail_index=train_data.edge_index[1],
        batch_size=1000,
        shuffle=True,
    )
    print("model loader done.")
    optimizer_map = {
        'transe': optim.Adam(model.parameters(), lr=0.01),
        'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
        'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
        'rotate': optim.Adam(model.parameters(), lr=1e-3),
    }
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) #optimizer_map[args.model]
    
    for epoch in range(1, args.epochs):
        loss = train(model, loader, optimizer)
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
            metrics = evaluate_link_prediction(model, valid_triples, all_triples, model.num_nodes)
            print(f'Epoch: {epoch:03d}, Val Mean Rank: {metrics["mean_rank"]:.2f}, '
                f'Val MRR: {metrics["mrr"]:.2f}, Val Hits@10: {metrics["hits@10"]:.2f}')

    print("One last test")
    metrics = evaluate_link_prediction(model, test_triples, all_triples)
    print(f'Test Mean Rank: {metrics["mean_rank"]:.2f}, '
                f'Test MRR: {metrics["mrr"]:.2f}, Test Hits@10: {metrics["hits@10"]:.2f}')

def train(model, loader, optimizer):
    model.train()
    #total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        pred = model.forward(head_index, rel_type)
        target = torch.zeros_like(pred)  # shape: [B, num_entities]
        target.scatter_(1, tail_index.view(-1, 1), 1.0)  # set 1 at true tail
        loss = model.loss(pred, target)
        loss.backward()
        optimizer.step()
        #total_loss += float(loss) * head_index.numel()
        #total_examples += head_index.numel()
    return loss #total_loss / total_examples


def data_to_triples(data: Data) -> List[Tuple[int, int, int]]:
    """
    Convert a PyG Data object with edge_index and edge_type into (head, relation, tail) triples.

    Args:
        data: PyG Data object with edge_index [2, num_edges] and edge_type [num_edges]

    Returns:
        List of (head, relation, tail) tuples
    """
    edge_index = data.edge_index
    edge_type = data.edge_type
    triples = []

    for i in range(edge_index.shape[1]):
        head = edge_index[0, i].item()
        tail = edge_index[1, i].item()
        rel = edge_type[i].item()
        triples.append((head, rel, tail))

    return triples

def generate_all_triples(train_triples, valid_triples=None, test_triples=None):
    """
    Combine train/valid/test triples into a single set for filtered evaluation.

    Args:
        train_triples: List of (head, relation, tail)
        valid_triples: Optional list of (head, relation, tail)
        test_triples: Optional list of (head, relation, tail)

    Returns:
        Set of all (head, relation, tail) triples
    """
    all_triples = set(train_triples)
    
    if valid_triples is not None:
        all_triples.update(valid_triples)

    if test_triples is not None:
        all_triples.update(test_triples)

    return all_triples

def evaluate_link_prediction(model, test_triples, all_triples, num_entities, device='cuda', hits_at_k=(1, 3, 10)):
    """
    Args:
        model: Trained ConvE model.
        test_triples: List of (head, relation, tail) test triples.
        all_triples: Set of all known triples for filtering (train + valid + test).
        num_entities: Total number of entities.
        device: torch device.
        hits_at_k: Tuple of k values for Hits@k.

    Returns:
        Dict with mean_rank, mrr, hits@k metrics.
    """
    model.eval()
    with torch.no_grad():
        ranks = []

        for head, relation, tail in tqdm(test_triples, desc="Evaluating"):
            head_tensor = torch.LongTensor([head]).to(device)
            rel_tensor = torch.LongTensor([relation]).to(device)

            # Compute scores for all possible tails
            scores = model(head_tensor, rel_tensor).squeeze(0)  # (num_entities,)

            # Filter: remove all known (h, r, t') except the correct one
            filt = set((head, relation, e) for e in range(num_entities)) - all_triples
            filt.add((head, relation, tail))  # add the ground truth back

            # Create a mask for filtered indices
            mask = torch.ones(num_entities, dtype=torch.bool).to(device)
            for (_, _, t) in filt:
                if t != tail:
                    mask[t] = 0
            scores = scores.masked_fill(~mask, float('-inf'))

            # Rank the true tail
            _, sorted_indices = torch.sort(scores, descending=True)
            rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        ranks = torch.tensor(ranks, dtype=torch.float)
        mean_rank = ranks.mean().item()
        mrr = (1.0 / ranks).mean().item()
        hits = {k: (ranks <= k).float().mean().item() for k in hits_at_k}

        return {
            "mean_rank": mean_rank,
            "mrr": mrr,
            **{f"hits@{k}": hits[k] for k in hits_at_k}
        }
    
if __name__ == "__main__":
    
    # read an argument from the command line, specifying the name of the dataset
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, required=True)
    #argparser.add_argument('--model', choices=model_map.keys(), type=str.lower, default='rotate')
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--testrate', type=int, default=10)
    argparser.add_argument('--saverate', type=int, default=10)
    args = argparser.parse_args()
    #mp.set_start_method('spawn', force=True)
    main(args)
