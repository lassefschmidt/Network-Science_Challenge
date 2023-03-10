# import own scripts
import util.preprocess_Data as prepData

# import external packages
import copy
import networkx as nx # graph data
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def train_val_split_pos_edges(G, edgelist, testing_ratio=0.2, seed=42):
    """
    generate pos edges for validation set, trim training graph respectively
    and ensure that it remains fully connected
    """
    # how many positive edges we want to sample for test data
    val_pos_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    val_pos_edges = pd.DataFrame(data = None, columns = edgelist.columns)

    # make a copy of the original graph
    G_train = copy.deepcopy(G)

    # start reducing the graph
    train_pos_edges = edgelist.loc[edgelist.y == 1]
    sampled, removed = 0, 0
    while (removed < val_pos_edges_num) and (sampled < val_pos_edges_num * 10):
        sampled += 1
        random_edge = random.sample(list(train_pos_edges.index.values), 1)[0] # sample one random edge
        u, v, _ = train_pos_edges.loc[random_edge].values # unpack edge
        
        if (G_train.degree(u) > 1 and G_train.degree(v) > 1): # only remove edge if both nodes have degree >1
            G_train.remove_edge(u, v)
            # check if this led to disconnected graph
            if not nx.is_connected(G_train):
                G_train.add_edge(u, v)
                continue
            train_pos_edges = train_pos_edges.drop(index = random_edge, inplace = False)
            val_pos_edges.loc[random_edge] = [u, v, 1]
            removed += 1
        else:
            continue
    
    # remove any isolated nodes (there should be none)
    G_train.remove_nodes_from(nx.isolates(G_train))

    # check that number of nodes has not changed
    node_num1 = G.number_of_nodes()
    node_num2 = G_train.number_of_nodes()
    assert node_num1 == node_num2

    # print key stats
    print(f"Number of positive edges for training: {len(train_pos_edges)}")
    print(f"Number of positive edges for validation: {len(val_pos_edges)}")
    print(f"Number of edges in original graph: {G.number_of_edges()}")
    print(f"Number of edges in training graph: {G_train.number_of_edges()}")

    return G_train, train_pos_edges, val_pos_edges

def train_val_split_neg_edges(edgelist, testing_ratio = 0.2, seed=42):
    """
    fetch negative edges for validation set from existing edgelist
    """
    # filter edgelist for negative samples
    edgelist = edgelist.loc[(edgelist.y == 0)]

    # perform train test split on it
    train_neg_edges, val_neg_edges = train_test_split(edgelist, test_size = testing_ratio, random_state = seed)

    return train_neg_edges, val_neg_edges

def split_frame(df):
    # split into X and y and drop node columns
    if "y" in df:
        y = df.loc[:, "y"]
        X = copy.deepcopy(df)
        X.drop(["node1", "node2", "y"], axis = 1, inplace = True)
        return X, y
    else:
        X = copy.deepcopy(df)
        X.drop(["node1", "node2"], axis = 1, inplace = True)
        return X

def load(testing_ratio = 0.2):
    """
    helper function that performs all loading + data cleaning (direct input for deep learning)
    """
    node_info = (pd.read_csv('data/node_information.csv', index_col = 0, header = None)
                 .rename_axis("node"))
    
    # read edge lists (train and test)
    trainval = pd.read_csv('data/train.txt', header = None, sep = " ").rename(columns = {0: "node1", 1: "node2", 2: "y"})
    test  = pd.read_csv('data/test.txt' , header = None, sep = " ").rename(columns = {0: "node1", 1: "node2"})

    # reindex all nodes (necessary for deep learning)
    node_idx_mapping = {old: new for new, old in node_info.reset_index()["node"].items()}

    node_info = (node_info
        .reset_index()
        .assign(node = lambda df_: [node_idx_mapping[node] for node in df_.node])
        .set_index("node")
    )
    trainval = (trainval
        .assign(node1 = lambda df_: [node_idx_mapping[node] for node in df_.node1])
        .assign(node2 = lambda df_: [node_idx_mapping[node] for node in df_.node2])
    )
    test = (test
        .assign(node1 = lambda df_: [node_idx_mapping[node] for node in df_.node1])
        .assign(node2 = lambda df_: [node_idx_mapping[node] for node in df_.node2])
    )

    # sort edge lists (so lower numbered node is always in first column)
    trainval = trainval[["node1", "node2"]].apply(lambda x: np.sort(x), axis = 1, raw = True).assign(y = trainval.y)
    test  = test[[ "node1", "node2"]].apply(lambda x: np.sort(x), axis = 1, raw = True)

    # remove edges from edgelist where source node == target node (always predict 1)
    trainval_tf = prepData.clean_edgelist(trainval)
    test_tf  = prepData.clean_edgelist(test)

    # build graph
    G = prepData.fetch_graph(trainval_tf)

    # generate train and validation data (postive edges)
    G_train, train_pos_edges, val_pos_edges = train_val_split_pos_edges(G, trainval_tf, testing_ratio = testing_ratio)

    # validate that graph is still connected
    prepData.get_gcc(G_train)

    # generate train and validation data (negative edges)
    train_neg_edges, val_neg_edges = train_val_split_neg_edges(trainval_tf, testing_ratio = testing_ratio)

    # append to dataframe
    train_tf = pd.concat([train_pos_edges, train_neg_edges]).sort_index()
    val_tf = pd.concat([val_pos_edges, val_neg_edges]).sort_index()

    return (G, G_train, node_info, train_tf, val_tf, trainval_tf, test, test_tf)

def load_transform(testing_ratio = 0.2):
    """
    helper function that performs all further pre-processsing necessary for classical ML approaches
    """
    (G, G_train, node_info, train_tf, val_tf, trainval_tf, test, test_tf) = load(testing_ratio)

    # enrich train and validation data
    print("Enriching train data...")
    train_tf = prepData.feature_extractor(train_tf, G_train, node_info)
    print("Enriching validation data...")
    val_tf   = prepData.feature_extractor(val_tf, G_train, node_info)
    # enrich test data
    print("Enriching test data...")
    test_tf = prepData.feature_extractor(test_tf, G, node_info, trainval = trainval_tf)
    
    # split
    X_train, y_train = split_frame(train_tf)
    X_val, y_val     = split_frame(val_tf)
    X_test           = split_frame(test_tf)

    # merge to get trainval data
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    return (G, G_train, node_info, train_tf, val_tf, trainval_tf, test, test_tf, X_train, y_train, X_val, y_val, X_trainval, y_trainval, X_test)