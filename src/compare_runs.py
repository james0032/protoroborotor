import json
from ast import literal_eval
from collections import defaultdict

files = [ ("CCD","results_900.txt"),
          ("CD","results_500.txt"),
          ("CDD","results_900.txt"),
          ("CGD","results_300.txt") ]

counts = defaultdict( dict )
MRR = defaultdict( dict )
hits = defaultdict( dict )

for directory, file in files:
    # resd the edge_map
    with open(f"robokop/{directory}/edge_map.json", "r") as f:
        edge_map = json.load(f)
    em = {}
    for edge, pnum in edge_map.items():
        e = {k:v for k,v in literal_eval(edge).items() if v != ""}
        em[pnum] = str(e)
    with open(f"robokop/{directory}/{file}", "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            x = line.strip().split("\t")
            if x[0] == "All":
                edge_label = "all"
            else:
                edge_label = em[f"predicate:{int(x[0])}"]
            counts[edge_label][directory] = int(x[1])
            MRR[edge_label][directory] = float(x[3])
            hits[edge_label][directory] = float(x[4])

for edge_label in counts:
    for d in counts[edge_label]:
        print(f"{edge_label}\t{d}\t{counts[edge_label][d]}\t{MRR[edge_label][d]}\t{hits[edge_label][d]}")
