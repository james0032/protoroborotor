import torch
import torch.nn as nn
import csv
import os
from torch_geometric.nn.models import RotatE

# === Config ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 100
PROJECTED_DIM = 512

BASE_PATH = '/workspace/data/robokop/rCD'
MODEL_PATH = os.path.join(BASE_PATH, 'model_10.pt')
ENTITY_OUTPUT_TSV = os.path.join(BASE_PATH, 'projected_entity_embeddings.tsv')

NODE_DICT_PATH = os.path.join(BASE_PATH, 'node_dict.tsv') #os.path.join(BASE_PATH, 'processed', 'node_dict.tsv')
REL_DICT_PATH = os.path.join(BASE_PATH, 'rel_dict.tsv') #os.path.join(BASE_PATH, 'processed', 'rel_dict.tsv')
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
    ).to(DEVICE)

    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# === Create projection layers ===

def create_projection_layers() -> tuple:
    entity_proj = nn.Linear(HIDDEN_DIM, PROJECTED_DIM).to(DEVICE)
    relation_proj = nn.Linear(HIDDEN_DIM, PROJECTED_DIM).to(DEVICE)
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
    entity_proj, relation_proj = create_projection_layers()

    # Only project and save entity embeddings for now
    entity_embed = model.entity_embedding.weight.detach()
    save_entity_embeddings_batched(entity_embed, node_dict, entity_proj, ENTITY_OUTPUT_TSV, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
