import pickle
import polars as pl
import torch
import torch.nn as nn
import os

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
# 2. Convert embeddings to tensor
emb_tensor = torch.tensor(newpl["topological_embedding"].to_list(), dtype=torch.float32, device=device)  # shape: [num_nodes, 100]

# 3. Apply projection
emb_proj = proj(emb_tensor)  # shape: [num_nodes, 512]

# 4. Convert back to list for Polars
newpl = newpl.with_columns(
    pl.Series("topological_embedding", emb_proj.cpu().detach().numpy().tolist())
)
newpl = newpl.lazy()

newpl_count = newpl.select(pl.len()).collect().row(0)[0]
print("New emb is ready for merge and has number of rows:", newpl_count.shape)
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