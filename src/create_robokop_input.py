# Given a base edges.jsonl from robokop, filter it and format the results to make the correct input for nn-geometric
# The input format is a tab-delimited file of subject\tpredicate\object\n.
import os

import jsonlines
import json

def remove_subclass_and_cid(edge, typemap):
    if edge["predicate"] == "biolink:subclass_of":
        return True
    if edge["subject"].startswith("CAID"):
        return True
    return False

def check_accepted(edge, typemap, accepted):
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in accepted:
        if acc[0] in subj_types and acc[1] in obj_types:
            return False
        if acc[1] in subj_types and acc[0] in obj_types:
            return False
    return True

def keep_CD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature") ]
    return check_accepted(edge, typemap, accepted)


def keep_CGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)


def keep_CDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity")]
    return check_accepted(edge, typemap, accepted)

def keep_CCDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCGGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)


def pred_trans(edge, edge_map):
    edge_key = {"predicate": edge["predicate"]}
    edge_key["subject_aspect_qualifier"] = edge.get("subject_aspect_qualifier", "")
    edge_key["object_aspect_qualifier"] = edge.get("object_aspect_qualifier", "")
    edge_key["subject_direction_qualifier"] = edge.get("subject_direction_qualifier", "")
    edge_key["object_direction_qualifier"] = edge.get("object_direction_qualifier", "")
    edge_key_string = json.dumps(edge_key, sort_keys=True)
    if edge_key_string not in edge_map:
        edge_map[edge_key_string] = f"predicate:{len(edge_map)}"
    return edge_map[edge_key_string]


def dump_edge_map(edge_map, outdir):
    output_file=f"{outdir}/edge_map.json"
    with open(output_file, "w") as writer:
        json.dump(edge_map, writer, indent=2)

def create_robokop_input(node_file="robokop/nodes.jsonl", edges_file="robokop/edges.jsonl", style="original"):
    outdir = f"robokop/{style}"
    output_file = f"{outdir}/rotorobo.txt"
    if style == "original":
        # This filters the edges by
        # 1) removing all subclass_of and
        # 2) removing all edges with a subject that starts with "CAID"
        remove_edge = remove_subclass_and_cid
    elif style == "CGD":
        # This keeps any edges between
        #   chemicals and genes
        #   genes and diseases
        #   chemicals and diseases
        # removes subclass edges
        remove_edge = keep_CGD
    elif style == "CD":
        # No subclasses
        # only chemical/disease edges
        remove_edge = keep_CD
    elif style == "CDD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CDD
    elif style == "CCD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CCD
    elif style == "CCGDD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CCGDD
    elif style == "CCGGDD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CCGGDD
    elif style == "CGGD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CGGD
    elif style == "CCDD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CCDD
    else:
        print("I don't know what you mean")
        return
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    type_map = {}
    with jsonlines.open(node_file) as reader:
        for node in reader:
            type_map[node["id"]] = set(node["category"])
    edge_map = {}
    with jsonlines.open(edges_file) as reader:
        with open(output_file, "w") as writer:
            for edge in reader:
                if remove_edge(edge,type_map):
                    continue
                writer.write(f"{edge['subject']}\t{pred_trans(edge,edge_map)}\t{edge['object']}\n")
    dump_edge_map(edge_map,outdir)

if __name__ == "__main__":
    #create_robokop_input(style="CCGDD")
    create_robokop_input(style="CCGGDD")
    print("CCGGDD created.")
    #create_robokop_input(style="CGGD")
    #create_robokop_input(style="CCD")
    #print("CCD created.")
    #create_robokop_input(style="CD")
    #print("CD created.")
    #create_robokop_input(style="CCDD")
    #print("CCDD created.")
    #create_robokop_input(style="CGD")
    #print("CGD created.")
    #create_robokop_input(style="original")
    #print("original created.")
