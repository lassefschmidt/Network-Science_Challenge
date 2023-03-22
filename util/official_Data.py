import pandas as pd
import numpy as np
from torch_geometric.datasets import Actor

def enrich_test():
    """
    ONLY FOR INTERNAL SCORING PURPOSES -- NOT TO BE USED FOR ANY SORT OF INFERENCE
    """
    node_info = (pd.read_csv('data/node_information.csv', index_col = 0, header = None)
                 .rename_axis("node"))
    
    # read edge lists (train and test)
    test  = pd.read_csv('data/test.txt' , header = None, sep = " ").rename(columns = {0: "node1", 1: "node2"})

    # sort edge lists (so lower numbered node is always in first column)
    test  = test[["node1", "node2"]].apply(lambda x: np.sort(x), axis = 1, raw = True)

    # enrich test list using official data
    actor = Actor(root = "data/actor")
    official_data = pd.DataFrame(actor.data.edge_index.T).rename(columns = {0: "node1", 1: "node2"})
    official_data = official_data[[ "node1", "node2"]].apply(lambda x: np.sort(x), axis = 1, raw = True).assign(y = 1).drop_duplicates()
    test = pd.merge(test, official_data, how = "left", on = ["node1", "node2"]).fillna(0).astype(int)

    # reindex all nodes (necessary for deep learning)
    node_idx_mapping = {old: new for new, old in node_info.reset_index()["node"].items()}

    return (test
        .assign(node1 = lambda df_: [node_idx_mapping[node] for node in df_.node1])
        .assign(node2 = lambda df_: [node_idx_mapping[node] for node in df_.node2])
    )