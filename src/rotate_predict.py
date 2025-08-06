import torch
import polars as pl
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.nn import RotatE
import json
import os
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
HIDDEN_DIM = 100

BASE_PATH = '/workspace/data/robokop/rCD'
MODEL_PATH = os.path.join(BASE_PATH, 'model_300.pt')
ENTITY_OUTPUT_TSV = os.path.join(BASE_PATH, 'rotate_predict.tsv')

NODE_DICT_PATH = os.path.join(BASE_PATH, 'processed', 'node_dict')
REL_DICT_PATH = os.path.join(BASE_PATH, 'processed', 'rel_dict')
BATCH_SIZE = 4096  # Tune based on memory size

TEST_FILE = os.path.join(BASE_PATH, "fold0_1k.tsv")
# ---------- Load Dictionaries ----------
# === Load index→name dictionaries from TSV ===

def load_dict_from_tsv(path: str) -> dict:
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            name, idx = line.strip().split('\t')
            mapping[name] = idx
    return mapping



# ---------- Load Model ----------

def load_trained_model(path: str, num_nodes: int, num_relations: int) -> RotatE:
    model = RotatE(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_channels=HIDDEN_DIM,
    ).to(DEVICE)

    checkpoint = torch.load(path, map_location=DEVICE)
    print("Checkpoint keys:", checkpoint.keys())
    # If your checkpoint has nested keys like "model_state_dict", extract them
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # fallback if it's already flat
    print("state_dict keys:", state_dict.keys())
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------- Convert to Batched PyTorch Dataset ----------
class TripleBatchIterableDataset(IterableDataset):
    def __init__(self, polars_lazy_df, batch_size=BATCH_SIZE):
        self.df = polars_lazy_df
        self.batch_size = batch_size

    def __iter__(self):
        df_materialized = self.df.collect(streaming=True)
        for batch in df_materialized.iter_rows(named=True):
            yield (
                torch.tensor(batch["head_id"], dtype=torch.long),
                torch.tensor(batch["rel_id"], dtype=torch.long),
                torch.tensor(batch["tail_id"], dtype=torch.long)
            )

# Convert string columns to integer IDs
def map_column_to_id(col_name, mapping):
    return pl.col(col_name).replace(mapping).cast(pl.Int64)


def main(args):
    node_dict = load_dict_from_tsv(NODE_DICT_PATH)
    rel_dict = load_dict_from_tsv(REL_DICT_PATH)

    model = load_trained_model(MODEL_PATH, len(node_dict), len(rel_dict))
    print("Model loaded!")


    # ---------- Polars Lazy Loading and String-to-ID Mapping ----------
    test_path = args.input  # tab-separated file: head \t relation \t tail

    lazy_df = pl.scan_csv(test_path, separator="\t", has_header=True).with_columns(
        pl.lit(70).alias("rel_id") # this is dynamic for each model trained. Need to parameterized this later!!
        )#, new_columns=["head", "tail"])



    lazy_df = lazy_df.with_columns([
        map_column_to_id("source", node_dict).alias("head_id"),
        #map_column_to_id(pl.col("rel"), rel_dict).alias("rel_id"),
        map_column_to_id("target", node_dict).alias("tail_id"),
    ]).select(["head_id", "rel_id", "tail_id"])




    dataset = TripleBatchIterableDataset(lazy_df)
    dataloader = DataLoader(dataset, batch_size=8192)


    # ---------- Run Predictions and Write Scores ----------
    with torch.no_grad(), open(os.path.join(BASE_PATH, args.output), "w") as fout:
        for batch in dataloader:
            h, r, t = [x.to(DEVICE) for x in batch]
            scores = model(h, r, t)  # shape: [batch_size]
            for s in scores.cpu().tolist():
                fout.write(f"{s}\n")
                
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    
    args = argparser.parse_args()
    main(args)