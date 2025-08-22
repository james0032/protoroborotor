import torch
import torch.nn as nn
import csv
import os
from torch_geometric.nn import RotatE
import argparse

# === Config ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 50
PROJECTED_DIM = 512

BASE_PATH = '/workspace/data/robokop/rCD'
MODEL_PATH = os.path.join(BASE_PATH, 'model_300.pt')
ENTITY_OUTPUT_TSV = os.path.join(BASE_PATH, 'projected_entity_embeddings.tsv')

NODE_DICT_PATH = os.path.join(BASE_PATH, 'processed', 'node_dict')
REL_DICT_PATH = os.path.join(BASE_PATH, 'processed', 'rel_dict')
BATCH_SIZE = 4096  # Tune based on memory size

# === Load index→name dictionaries from TSV ===

def load_dict_from_tsv(path: str) -> dict:
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            name, idx = line.strip().split('\t')
            mapping[int(idx)] = name
    return mapping


# === Load trained RotatE model ===

def load_trained_model(path: str, num_nodes: int, num_relations: int) -> RotatE:
    model = RotatE(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_channels=HIDDEN_DIM,
    ).to("cpu")

    checkpoint = torch.load(path, map_location="cpu")
    print("Checkpoint keys:", checkpoint.keys())
    # If your checkpoint has nested keys like "model_state_dict", extract them
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # fallback if it's already flat
    print("state_dict keys:", state_dict.keys())
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# === Create projection layers ===

def create_projection_layers() -> tuple:
    entity_proj = nn.Linear(2*HIDDEN_DIM, PROJECTED_DIM).to(DEVICE)
    relation_proj = nn.Linear(2*HIDDEN_DIM, PROJECTED_DIM).to(DEVICE)
    return entity_proj, relation_proj


# === Save projected entity embeddings in batch ===

def save_entity_embeddings_batched(
    entity_embed: torch.Tensor,
    index_to_name: dict,
    projection_layer: nn.Module,
    path: str,
    batch_size: int = 4096
):
    entity_embed = entity_embed.to(DEVICE)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        for start in range(0, entity_embed.size(0), batch_size):
            end = min(start + batch_size, entity_embed.size(0))
            batch_tensor = entity_embed[start:end]
            projected = projection_layer(batch_tensor).detach().cpu()

            for i, emb in enumerate(projected):
                idx = start + i
                name = index_to_name.get(idx, f"unknown_{idx}")
                emb_str = "[" + ", ".join(f"{x:.3f}" for x in emb.tolist()) + "]"
                writer.writerow([name, emb_str])

    print(f"Saved {entity_embed.size(0)} projected entity embeddings to: {path}")


# === Main ===

def main():
    print(f"Using device: {DEVICE}")

    node_dict = load_dict_from_tsv(NODE_DICT_PATH)
    rel_dict = load_dict_from_tsv(REL_DICT_PATH)

    NUM_ENTITIES = len(node_dict)
    NUM_RELATIONS = len(rel_dict)

    model = load_trained_model(MODEL_PATH, NUM_ENTITIES, NUM_RELATIONS)
    print("Model loaded!")
    entity_proj, relation_proj = create_projection_layers()

    # Only project and save entity embeddings for now
    entity_real = model.node_emb.weight.detach()
    entity_imag = model.node_emb_im.weight.detach()
    entity_embed = torch.cat([entity_real, entity_imag], dim=1)  # [num_entities, 2 * hidden_dim]

    #relation_embed = model.rel_emb.weight.detach()
    
    save_entity_embeddings_batched(entity_embed, node_dict, entity_proj, ENTITY_OUTPUT_TSV, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, required=True)
    
    main()
