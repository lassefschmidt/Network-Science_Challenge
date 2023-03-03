# parse & handle data
import networkx as nx # graph data
import numpy as np
from numpy.linalg import norm

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

def get_gcc(graph):
    """
    check if graph is connected -- if not, return greatest connected component subgraph
    """
    # Is the given graph connected?
    connected = nx.is_connected(graph) # check if the graph is connected or not
    if connected:
        print("The graph is connected")
        return graph
    
    print("The graph is not connected")
    
    # Find the number of connected components
    num_of_cc = nx.number_connected_components(graph)
    print("Number of connected components: {}".format(num_of_cc))
    
    # Get the greatest connected component subgraph
    gcc_nodes = max(nx.connected_components(graph), key=len)
    gcc = graph.subgraph(gcc_nodes)
    node_fraction = gcc.number_of_nodes() / float(graph.number_of_nodes())
    edge_fraction = gcc.number_of_edges() / float(graph.number_of_edges())
    
    print("Fraction of nodes in GCC: {:.3f}".format(node_fraction))
    print("Fraction of edges in GCC: {:.3f}".format(edge_fraction))

    return gcc

def enrich_edgelist(edgelist, graph, node_info):
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
    RA  = transform_generator_to_dict(nx.resource_allocation_index(graph, ebunch))
    JCC = transform_generator_to_dict(nx.jaccard_coefficient(graph, ebunch))
    AA  = transform_generator_to_dict(nx.adamic_adar_index(graph, ebunch))
    PA  = transform_generator_to_dict(nx.preferential_attachment(graph, ebunch))
    CNC = transform_generator_to_dict(nx.common_neighbor_centrality(graph, ebunch))

    # append new columns
    return (edgelist
        .assign(RA  = lambda df_: [RA[(u, v)]  for u, v in zip(df_.node1, df_.node2)])
        .assign(JCC = lambda df_: [JCC[(u, v)] for u, v in zip(df_.node1, df_.node2)])
        .assign(AA  = lambda df_: [AA[(u, v)]  for u, v in zip(df_.node1, df_.node2)])
        .assign(PA  = lambda df_: [PA[(u, v)]  for u, v in zip(df_.node1, df_.node2)])
        .assign(PA_log = lambda df_: np.log(df_.PA))
        .assign(CNC = lambda df_: [CNC[(u, v)] for u, v in zip(df_.node1, df_.node2)])
        .assign(CNC_log = lambda df_: np.log(df_.CNC))
        .assign(CS  = lambda df_: [cosine_similarity(node_info.loc[u], node_info.loc[v]) for u, v in zip(df_.node1, df_.node2)])
    )