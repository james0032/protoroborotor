import polars as pl
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os

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
newpl = pl.scan_csv(os.path.join(BASE_PATH, "projected_entity_embeddings.tsv"), separator='\t', has_header=False)
schema = newpl.collect_schema()
col_names = schema.names()

newpl = newpl.rename({col_names[0]: "id", col_names[1]: "topological_embedding"})
newpl = newpl.with_columns(
    pl.col("id").cast(pl.Utf8).str.replace_all(r"\s+", ""),
    pl.col("topological_embedding")
      .str.strip_chars("[]")                          # remove [ ]
      .str.split(",")                                 # split into list of strings
      .list.eval(pl.element().str.strip_chars().cast(pl.Float32))  # trim & cast
)
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

output_dir = os.path.join(BASE_PATH, "rotate_emb")
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