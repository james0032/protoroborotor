import polars as pl
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import os

BASE_PATH = "/workspace/data/robokop/rCD"

df = pl.read_parquet("gs://mtrx-us-central1-hub-dev-storage/kedro/data/tests/rotate_esuite/runs/rtx-base-feat3-3ce2b620/datasets/embeddings/feat/nodes_with_embeddings/")
print("Begining size of nodes with embeddings", df.shape)
new = pd.read_csv(os.path.join(BASE_PATH, "projected_entity_embeddings.tsv"), sep='\t', header=None, names=["id", "topo"])

# Step 2: Vectorized string-to-list[f32] parsing
new["topological_embedding"] = new["topo"].progress_apply(
    lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
)
# Step 3: Convert to Polars and cast (if needed)
new = pl.from_pandas(new)
new = new.with_columns(
    pl.col("topological_embedding").cast(pl.List(pl.Float32))
)
new = new.drop("topo")

# Step 4: df to drop topo and join new to df
df = df.drop("topological_embedding")
df = df.join(new, on="id", how="inner")
print("Final size of nodes with embeddings", df.shape)

output_dir = os.path.join(BASE_PATH, "rotate_emb")
os.makedirs(output_dir, exist_ok=True)

# Split into 200 roughly equal partitions
num_partitions = 200
partition_size = len(new) // num_partitions + 1

for i in range(num_partitions):
    start = i * partition_size
    end = min((i + 1) * partition_size, len(new))
    df_slice = df.slice(start, end - start)
    
    # Save each partition
    df_slice.write_parquet(
        f"{output_dir}/part_{i:05d}.snappy.parquet", 
        compression="snappy"
    )