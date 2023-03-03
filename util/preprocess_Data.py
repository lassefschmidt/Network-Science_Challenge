# parse & handle data
import networkx as nx # graph data
import copy
import random
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import pandas as pd

def clean_edgelist(edgelist):
    """
    Remove edges from edgelist where source node == target node
    """
    return edgelist.loc[(edgelist.node1 != edgelist.node2)]

def fetch_graph(edgelist):
    """
    Create a graph based on an edgelist. Make sure that it doesn't contain edges where source node == target node
    """
    if "y" in edgelist:
        edgelist = edgelist.loc[(edgelist.y == 1)]
    
    return nx.from_pandas_edgelist(edgelist, "node1", "node2")

def get_gcc(G):
    """
    check if graph is connected -- if not, return greatest connected component subgraph
    """
    # Is the given graph connected?
    connected = nx.is_connected(G) # check if the graph is connected or not
    if connected:
        print("The graph is connected")
        return G
    
    print("The graph is not connected")
    
    # Find the number of connected components
    num_of_cc = nx.number_connected_components(G)
    print("Number of connected components: {}".format(num_of_cc))
    
    # Get the greatest connected component subgraph
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes)
    node_fraction = gcc.number_of_nodes() / float(G.number_of_nodes())
    edge_fraction = gcc.number_of_edges() / float(G.number_of_edges())
    
    print("Fraction of nodes in GCC: {:.3f}".format(node_fraction))
    print("Fraction of edges in GCC: {:.3f}".format(edge_fraction))

    return gcc

def enrich_edgelist(edgelist, G, node_info):
    """
    Enrich edgelist with graph-based edge features
    (e.g. resource allocation index, jaccard coefficient, etc.)
    and similarity metrics based on node-level keyword embedding
    """
    # helper function to transform networkx generator objects
    def transform_generator_to_dict(generator_obj):
        result = dict()
        for (u, v, value) in generator_obj:
            result[(u, v)] = value
        return result
    
    # helper function to compute cosine similarity of keyword embeddings
    def cosine_similarity(emb1, emb2):
        return np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    
    # compute graph-based edge features
    ebunch = [(u, v) for u, v in zip(edgelist.node1, edgelist.node2)]
    RA  = transform_generator_to_dict(nx.resource_allocation_index(G, ebunch))
    JCC = transform_generator_to_dict(nx.jaccard_coefficient(G, ebunch))
    AA  = transform_generator_to_dict(nx.adamic_adar_index(G, ebunch))
    PA  = transform_generator_to_dict(nx.preferential_attachment(G, ebunch))
    CNC = transform_generator_to_dict(nx.common_neighbor_centrality(G, ebunch))

    # append new columns
    return (edgelist
        .assign(RA  = lambda df_: [RA[(u, v)]  for u, v in zip(df_.node1, df_.node2)])
        .assign(JCC = lambda df_: [JCC[(u, v)] for u, v in zip(df_.node1, df_.node2)])
        .assign(AA  = lambda df_: [AA[(u, v)]  for u, v in zip(df_.node1, df_.node2)])
        .assign(PA  = lambda df_: [PA[(u, v)]  for u, v in zip(df_.node1, df_.node2)])
        .assign(PA_log = lambda df_: np.log(df_.PA))
        #.assign(CNC = lambda df_: [CNC[(u, v)] for u, v in zip(df_.node1, df_.node2)]) # feature leads to huge overfit! (obvious, because we use the neighbors here directly!)
        #.assign(CNC_log = lambda df_: np.log(df_.CNC)) # feature leads to huge overfit! (obvious, because we use the neighbors here directly!)
        .assign(nodeInfo_CS    = lambda df_: [cosine_similarity(node_info.loc[u], node_info.loc[v]) for u, v in zip(df_.node1, df_.node2)])
        .assign(nodeInfo_diff  = lambda df_: [sum(abs(node_info.loc[u] - node_info.loc[v])) for u, v in zip(df_.node1, df_.node2)])
    )

def train_val_split_pos_edges(G, testing_ratio=0.2, seed=42):
    """
    generate pos edges for validation set, trim training graph respectively
    and ensure that it remains fully connected
    """
    # how many positive edges we want to sample for test data
    val_pos_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    val_pos_edges = []

    # make a copy of the original graph
    G_train = copy.deepcopy(G)

    # start reducing the graph
    sampled, removed = 0, 0
    while (removed < val_pos_edges_num) and (sampled < val_pos_edges_num * 10):
        sampled += 1
        random_edge = random.sample(G_train.edges, 1) # sample one random edge
        [(u, v)] = random_edge # unpack edge
        if (G_train.degree(u) > 1 and G_train.degree(v) > 1): # only remove edge if both nodes have degree >1
            G_train.remove_edge(u, v)
            # check if this led to disconnected graph
            if not nx.is_connected(G_train):
                G_train.add_edge(u, v)
                continue
            val_pos_edges.append((u, v, 1))
            removed += 1
        else:
            continue
    
    # remove any isolated nodes (there should be none)
    G_train.remove_nodes_from(nx.isolates(G_train))

    # extract remaining edges from G_train (that's our training data)
    train_pos_edges = [(u, v, 1) for (u, v) in G_train.edges()]

    # convert both lists into DataFrame
    train_pos_edges = pd.DataFrame(train_pos_edges).rename(columns = {0: "node1", 1: "node2", 2: "y"})
    val_pos_edges = pd.DataFrame(val_pos_edges).rename(columns = {0: "node1", 1: "node2", 2: "y"})

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

def load_prep_data():
    """
    helper function that performs all loading + preprocessing for model building
    """
    node_info = (pd.read_csv('data/node_information.csv', index_col = 0, header = None)
                 .rename_axis("node"))
    
    # read edge lists (train and test)
    train = pd.read_csv('data/train.txt', header = None, sep = " ").rename(columns = {0: "node1", 1: "node2", 2: "y"})
    test  = pd.read_csv('data/test.txt' , header = None, sep = " ").rename(columns = {0: "node1", 1: "node2"})

    # sort edge lists (so lower numbered node is always in first column)
    train = train[["node1", "node2"]].apply(lambda x: np.sort(x), axis = 1, raw = True).assign(y = train.y)
    test  = test[[ "node1", "node2"]].apply(lambda x: np.sort(x), axis = 1, raw = True)

    # remove edges from edgelist where source node == target node (always predict 1)
    train_tf = clean_edgelist(train)
    test_tf  = clean_edgelist(test)

    # build graph
    G = fetch_graph(train_tf)

    # generate train and validation data (postive edges)
    G_train, train_pos_edges, val_pos_edges = train_val_split_pos_edges(G, testing_ratio = 0.1)

    # validate that graph is still connected
    get_gcc(G_train)

    # generate train and validation data (negative edges)
    train_neg_edges, val_neg_edges = train_val_split_neg_edges(train_tf, testing_ratio = 0.1)

    # append to dataframe
    train_tf = pd.concat([train_pos_edges, train_neg_edges])
    val_tf = pd.concat([val_pos_edges, val_neg_edges])

    # enrich train and validation data
    train_tf = enrich_edgelist(train_tf, G_train, node_info)
    val_tf   = enrich_edgelist(val_tf, G_train, node_info)

    # enrich test data
    test_tf  = enrich_edgelist(test_tf, G, node_info)

    # split
    X_train, y_train = split_frame(train_tf)
    X_val, y_val     = split_frame(val_tf)
    X_test           = split_frame(test_tf)

    return G, G_train, train_tf, val_tf, test, test_tf, X_train, y_train, X_val, y_val, X_test