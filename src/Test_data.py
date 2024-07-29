# This code is a very light remodeling of nn-geometric's FB15k_237
# See https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/freebase.py

from typing import Callable, Dict, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from collections import defaultdict


class PredicateTestData(InMemoryDataset):
    # An in-memory data set for test data that breaks up the data by predicate for evaluation
    def __init__(
        self,
        root: str,
        split: int = 0,
        num_preds: int = 1,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.num_preds = num_preds

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)


        #Find the right path for this predicate and load it.
        path = self.processed_paths[split]
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        # All of the data for this is in the test file, we're going to split it out
        return ['robo_test.txt' for i in range(self.num_preds)]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{i}.pt' for i in range(self.num_preds)]

    def process(self) -> None:
        # read the raw file, and break it into pieces by predicate, save it
        data_list: Dict[str, Data] = {}
        node_dict: Dict[str, int] = {}
        rel_dict: Dict[str, int] = {}

        # There's only one raw path
        path = self.raw_paths[0]
        subgraphs = defaultdict(list)
        with open(path) as f:
            all_lines = [x.split('\t') for x in f.read().split('\n')[:-1]]

        for line in all_lines:
            subgraphs[line[1]].append(line)

        for predicate,lines in subgraphs.items():
            edge_index = torch.empty((2, len(lines)), dtype=torch.long)
            edge_type = torch.empty(len(lines), dtype=torch.long)
            for i, (src, rel, dst) in enumerate(lines):
                if src not in node_dict:
                    node_dict[src] = len(node_dict)
                if dst not in node_dict:
                    node_dict[dst] = len(node_dict)
                if rel not in rel_dict:
                    rel_dict[rel] = len(rel_dict)

                edge_index[0, i] = node_dict[src]
                edge_index[1, i] = node_dict[dst]
                edge_type[i] = rel_dict[rel]

            data = Data(edge_index=edge_index, edge_type=edge_type)
            # the predicate will be something like "predicate:7"
            pnum = predicate.split(":")[1]
            data.num_nodes = len(node_dict)
            data_list[pnum]=data


        for path in self.processed_paths:
            pnum = path.split("/")[-1].split(".")[0]
            if pnum in data_list:
                self.save([data_list[pnum]], path)