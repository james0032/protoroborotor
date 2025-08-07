import json

# Load entity IDs from index map
def load_index_map(file_path):
    index_map_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                index_map_ids.add(parts[0])
    return index_map_ids

# Load entity IDs from JSONL
def load_jsonl_ids(file_path):
    jsonl_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            jsonl_ids.append(obj["id"])
    return jsonl_ids

# Find unique entities in JSONL but not in index map
def find_unique_entities(jsonl_ids, index_map_ids):
    return [eid for eid in jsonl_ids if eid not in index_map_ids]

# Main logic
if __name__ == "__main__":
    index_map_file = "/workspace/data/robokop/rCD/processed/node_dict"     # Replace with your actual file path
    jsonl_file = "/workspace/data/robokop/nodes.jsonl"        # Replace with your actual file path

    index_map_ids = load_index_map(index_map_file)
    jsonl_ids = load_jsonl_ids(jsonl_file)

    unique_entities = find_unique_entities(jsonl_ids, index_map_ids)

    print(f"Number of unique entities in JSONL: {len(unique_entities)}")
    print("Top 60 unique entities:")
    for eid in unique_entities[:60]:
        print(f'\{"object":{eid},"subject":{eid},"predicate":"biolink:subclass_of"\}')
        #print(eid)