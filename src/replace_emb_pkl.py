import pickle
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_PATH = "/workspace/data/robokop/rCD"

# Step 1: Read both old and new emb files
df = pl.scan_parquet("gs://mtrx-us-central1-hub-dev-storage/data/01_RAW/modeling/UNC10/rotatE_on_KG2/rotate_emb_match_feat3/")
#print("Begining size of nodes with embeddings", df.shape)
df = df.with_columns(pl.col("id").cast(pl.Utf8).str.strip_chars('"'))
row_count = df.select(pl.len()).collect().row(0)[0]
print("df has number of rows", row_count)
dupes = df.group_by("id").agg(pl.len().alias("count")).filter(pl.col("count")>1)
print("Duplicated id in matrix pipeline generated embedding file", dupes.collect())

# Step 2: Read new embedding file and format data type to match matrix pipeline
# 1. Load pickle file
with open("/projects/aixb/jchung/ROBOKOP/git/keep_all_nobert/xDTD_training_pipeline/data/text_embedding/embedding_biobert_namecat.pkl", "rb") as f:
    data = pickle.load(f)  # data is a dict {id: list[f32]}

# 2. Convert dict to Polars DataFrame
newpl = pl.DataFrame({
    "id": list(data.keys()),
    "topological_embedding": list(data.values())
})
proj = nn.Linear(100, 512).to(device)

batch_size = 1024  # adjust based on your GPU memory
emb_list = newpl["topological_embedding"].to_list()
num_nodes = len(emb_list)
print(f"pkl has number of nodes:{num_nodes}")
num_batches = math.ceil(num_nodes / batch_size)


pl_chunks = []

for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, num_nodes)
    
    batch_tensor = torch.tensor(emb_list[start:end], dtype=torch.float32, device=device)
    batch_proj = proj(batch_tensor)
    print(f"batch {i:3d} projection done.")
    batch_proj_cpu = batch_proj.cpu().detach().numpy().tolist()
    pl_chunk = pl.DataFrame({
        "id": newpl["id"][start:end],
        "topological_embedding": batch_proj_cpu
    })
    pl_chunks.append(pl_chunk)
    #all_proj.append(batch_proj.cpu().detach().numpy())

# Concatenate at the end (Polars is memory-efficient here)
newpl = pl.concat(pl_chunks)
newpl.write_parquet("topological_embeddings_512.parquet")
print("Parquet file saved successfully.")
newpl = newpl.lazy()

newpl_count = newpl.select(pl.len()).collect().row(0)[0]
print("New emb is ready for merge and has number of rows:", newpl_count)
dupes = newpl.group_by("id").agg(pl.len().alias("count")).filter(pl.col("count")>1)
print("Duplicate ids in new embedding", dupes.collect())

# Step 3: df to drop topo and join new to df
df = df.drop("topological_embedding")
df = df.join(newpl, on="id", how="left")
#print("Final size of nodes with embeddings", df.shape)
row_count = df.select(pl.len()).collect().row(0)[0]
print("After join, df has number of rows", row_count)
nullcheck = df.filter(pl.col("topological_embedding").is_null())
print(nullcheck.collect())
output_dir = os.path.join(BASE_PATH, "biobert_emb")
os.makedirs(output_dir, exist_ok=True)

# Split into 200 roughly equal partitions
num_partitions = 200

partition_size = (row_count // num_partitions) + 1
df.sink_parquet(f"{output_dir}/all.snappy.parquet", compression="snappy")
#for i in range(num_partitions):
#    start = i * partition_size
#    length = min(partition_size, row_count-start)
#    df_slice = df.slice(start, length)
#    print(f"Writing partition {i}, rows {start} to {start + length -1}")
#    # Save each partition
#    df_slice.sink_parquet(
#        f"{output_dir}/part_{i:05d}.snappy.parquet", 
#        compression="snappy"
#    )