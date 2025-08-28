import torch
import torch.nn as nn
import csv
import os
import argparse
import polars as pl

# === Config ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 512
PROJECTED_DIM = 256

BASE_PATH = '/workspace/data/robokop/rCD'
#MODEL_PATH = os.path.join(BASE_PATH, 'model_300.pt')
ENTITY_OUTPUT_TSV = os.path.join(BASE_PATH, 'projected_entity_embeddings.tsv')

NODE_DICT_PATH = os.path.join(BASE_PATH, 'processed', 'node_dict')
REL_DICT_PATH = os.path.join(BASE_PATH, 'processed', 'rel_dict')
BATCH_SIZE = 4096  # Tune based on memory size


# Load n2v emb that has 512-dim
df = pl.scan_parquet("gs://mtrx-us-central1-hub-dev-storage/kedro/data/tests/emb_replace_robokop/datasets/embeddings/feat/nodes_with_embeddings/")
project = nn.Linear(INPUT_DIM, PROJECTED_DIM).to(DEVICE)
project.eval()

# --- 3. Collect in batches and project ---
BATCH_SIZE = 100000
result_batches = []

for df_batch in df.collect().iter_slices(n_rows=BATCH_SIZE):
    # Convert list column -> torch tensor
    x = torch.tensor(df_batch["topological_embedding"].to_list(), dtype=torch.float32).to(DEVICE)  # (B, 512)

    # Project with torch
    with torch.no_grad():
        y = project(x).to("cpu").numpy().tolist()  # list[list[float]] shape (B, 256)

    # Insert back into Polars as list[f32]
    df_batch = df_batch.with_columns(
        pl.Series("embedding_proj", y).cast(pl.List(pl.Float32))
    )

    result_batches.append(df_batch)

# --- 4. Concatenate batches back together ---
df = pl.concat(result_batches)

# --- 5. Replace old embedding if desired ---
df = df.drop("topological_embedding").rename({"embedding_proj": "topological_embedding"})

df.write_parquet("/workspace/data/robokop/all.nodes.emb.snappy.parquet", separator='\t')