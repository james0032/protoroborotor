# Given a base edges.jsonl from robokop, filter it and format the results to make the correct input for nn-geometric
# The input format is a tab-delimited file of subject\tpredicate\object\n.

import jsonlines
import json

def remove_edge(edge):
    if edge["predicate"] == "biolink:subclass_of":
        return True
    if edge["subject"].startswith("CAID"):
        return True
    return False


def pred_trans(edge, edge_map):
    edge_key = {"predicate": edge["predicate"]}
    edge_key["subject_aspect_qualifier"] = edge.get("subject_aspect_qualifier", "")
    edge_key["object_aspect_qualifier"] = edge.get("object_aspect_qualifier", "")
    edge_key["subject_direction_qualifier"] = edge.get("subject_direction_qualifier", "")
    edge_key["object_direction_qualifier"] = edge.get("object_directoin_qualifier", "")
    edge_key_string = json.dumps(edge_key, sort_keys=True)
    if edge_key_string not in edge_map:
        edge_map[edge_key_string] = f"predicate:{len(edge_map)}"
    return edge_map[edge_key_string]


def dump_edge_map(edge_map, output_file="robokop/edge_map.json"):
    with open(output_file, "w") as writer:
        json.dump(edge_map, writer, indent=2)
def create_robokop_input(edges_file="robokop/edges.jsonl", output_file="robokop/rotorobo.txt"):
    edge_map = {}
    with jsonlines.open(edges_file) as reader:
        with open(output_file, "w") as writer:
            for edge in reader:
                if remove_edge(edge):
                    continue
                writer.write(f"{edge['subject']}\t{pred_trans(edge,edge_map)}\t{edge['object']}\n")
    dump_edge_map(edge_map)

if __name__ == "__main__":
    create_robokop_input()
    print("robokop/rotorobo.txt created.")