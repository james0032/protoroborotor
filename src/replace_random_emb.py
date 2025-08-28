import polars as pl
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os

BASE_PATH = "/workspace/data/robokop/rCD_robokop_emb_predicate_only"
DIM = 512
# Step 1: Read both old emb files
df = pl.scan_parquet("gs://mtrx-us-central1-hub-dev-storage/kedro/data/tests/emb_replace_robokop/datasets/embeddings/feat/nodes_with_embeddings/")
#print("Begining size of nodes with embeddings", df.shape)
df = df.with_columns(pl.col("id").cast(pl.Utf8).str.strip_chars('"'))
row_count = df.select(pl.len()).collect().row(0)[0]
print("df has number of rows", row_count)
#dupes = df.group_by("id").agg(pl.len().alias("count")).filter(pl.col("count")>1)
#print("Duplicated id in matrix pipeline generated embedding file", dupes.collect())

# Step 2: collect all IDs 
all_ids = (
    df.select("topological_embedding")   # pick the column
      .collect()                         # trigger execution
      .to_series()                       # convert to Series
      .to_list()                         # finally Python list
)

print(f"There are {len(all_ids)} nodes has no embeddings from projected_entity_embeddings.tsv")


# Step 5: make random vectors for those IDs
np.random.seed(42)
rand_vectors = [np.random.rand(DIM).astype(np.float64).tolist() for _ in all_ids]

rand_df = pl.DataFrame({
    "id": all_ids,
    "topological_embedding": rand_vectors
})
# Step 6: join back
df.drop(["topological_embedding"])
df = df.join(rand_df.lazy(), on="id", how="left")

# Step 7: Check again if there is any null embeddings in topological_embedding column
nullcheck = df.filter(pl.col("topological_embedding").is_null())
print(nullcheck.collect())

output_dir = os.path.join(BASE_PATH, "random_emb")
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