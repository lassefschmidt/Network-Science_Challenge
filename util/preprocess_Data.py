# parse & handle data
import networkx as nx # graph data
from node2vec import Node2Vec
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

def feature_extractor(edgelist, G, node_info):
    """
    Enrich edgelist with graph-based edge features
    (e.g. resource allocation index, jaccard coefficient, etc.)
    and similarity metrics based on node-level keyword embedding

    Features that didn't work out: HITS algorithm, eigenvector/katz/common-neighbor/load centrality, voterank, CF/SCF enhanced RA (huge overfit), dispersion
    """
    # helper function to transform networkx generator objects into feature dicts
    def transform_generator_to_dict(generator_obj):
        result = dict()
        for (u, v, value) in generator_obj:
            result[(u, v)] = value
        return result
    
    # helper function to get CF- and SCF-enhanced features (see https://doi.org/10.1016/j.physa.2021.126107)
    def enhance_CF(edge, feature_dict, feature_func):
        (u, v) = edge
        # get neighbors of each node
        neighbors_u = [(n, v) for n in G.neighbors(u) if n != v]
        neighbors_v = [(n, u) for n in G.neighbors(v) if n != u]
        # compute similarity of neighbors of source (target) with target (source)
        sim_neighbors_u_to_v = sum([get_sim(edge, feature_dict, feature_func) for edge in neighbors_u])
        sim_neighbors_v_to_u = sum([get_sim(edge, feature_dict, feature_func) for edge in neighbors_v])
        return sim_neighbors_u_to_v + sim_neighbors_v_to_u
    
    def enhance_SCF(edge, feature_dict, feature_func):
        (u, v) = edge
        sim_neighbors = enhance_CF(edge, feature_dict, feature_func)
        sim_edge = sum([get_sim(edge, feature_dict, feature_func) for edge in [(u,v), (v,u)]])
        return sim_neighbors + sim_edge
    
    def get_sim(edge, feature_dict, feature_func):
        if edge not in feature_dict:
            feature_dict[edge] = sum([value for (_, _, value) in feature_func(G, [edge])])
        return feature_dict[edge]

    # helper function to compute cosine similarity of keyword embeddings
    def cosine_similarity(emb1, emb2):
        return np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    
    # compute graph-based node features
    DCT = nx.degree_centrality(G)
    BCT = nx.betweenness_centrality(G)
    # compute graph-based edge features
    ebunch = [(u, v) for u, v in zip(edgelist.node1, edgelist.node2)]
    RA  = transform_generator_to_dict(nx.resource_allocation_index(G, ebunch))
    JCC = transform_generator_to_dict(nx.jaccard_coefficient(G, ebunch))
    AA  = transform_generator_to_dict(nx.adamic_adar_index(G, ebunch))
    PA  = transform_generator_to_dict(nx.preferential_attachment(G, ebunch))

    # append new columns
    return (edgelist
        # node_info features
        .assign(nodeInfo_CS    = lambda df_: [cosine_similarity(node_info.loc[u], node_info.loc[v]) for u, v in zip(df_.node1, df_.node2)])
        .assign(nodeInfo_diff  = lambda df_: [sum(abs(node_info.loc[u] - node_info.loc[v])) for u, v in zip(df_.node1, df_.node2)])
        # node features
        .assign(source_DCT  = lambda df_: [DCT[node] for node in df_.node1])
        .assign(target_DCT  = lambda df_: [DCT[node] for node in df_.node2])
        .assign(BCT_diff    = lambda df_: [BCT[v]- BCT[u] for u, v in zip(df_.node1, df_.node2)])
        # edge features
        .assign(RA     = lambda df_: [RA[edge]  for edge in zip(df_.node1, df_.node2)])
        .assign(JCC    = lambda df_: [JCC[edge] for edge in zip(df_.node1, df_.node2)])
        .assign(AA     = lambda df_: [AA[edge]  for edge in zip(df_.node1, df_.node2)])
        .assign(PA     = lambda df_: [PA[edge]  for edge in zip(df_.node1, df_.node2)])
        .assign(CF_PA  = lambda df_: [enhance_CF(edge,  PA, nx.preferential_attachment) for edge in zip(df_.node1, df_.node2)])
        .assign(SCF_PA = lambda df_: [enhance_SCF(edge, PA, nx.preferential_attachment) for edge in zip(df_.node1, df_.node2)])
        .assign(PA_log = lambda df_: np.log(df_.PA))
        .assign(CF_PA_log = lambda df_: np.log(df_.CF_PA))
    )