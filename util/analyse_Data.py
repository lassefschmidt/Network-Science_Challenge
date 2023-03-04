# parse & handle data
import networkx as nx # graph data
import numpy as np

# visualization
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def compute_network_characteristics(graph):
    """
    compute key statistics of a given networkx.graph instance
    """
    prop = {}
    prop['N'] = graph.number_of_nodes() # number of nodes
    prop['M'] = graph.number_of_edges() # number of edges
    # degrees = list(dict(G.degree()).values())
    degrees = [degree for node, degree in graph.degree()] # degree list
    prop['min_degree'] = np.min(degrees) # minimum degree
    prop['max_degree'] = np.max(degrees) # maximum degree
    prop['mean_degree'] = np.mean(degrees) # mean of node degrees
    prop['median_degree'] = np.median(degrees) # median of node degrees
    prop['density'] = nx.density(graph) # density of the graph
    prop['avg_clustering'] = nx.average_clustering(graph) # average clustering coeff
    prop['avg_shortest_path'] = nx.average_shortest_path_length(graph)
    return prop

def invert_dict_count(X_dict, y_dict = None, agg = "count"):
    """
    invert a given dict and potentially replace the inverted values (keys of X_dict) with values from another dict (values of y_dict)
    afterwards aggegrate the inverted values and return a tuple (keys, values) sorted by ascending keys
    """
    # invert
    inv_map = {}
    for k, v in X_dict:
        inv_map[v] = inv_map.get(v, []) + [k]
        
    if y_dict is not None:
        for v, k_lst in inv_map.items():
            for idx, k in enumerate(k_lst):
                inv_map[v][idx] = y_dict[k]
    
    # generate agg func
    if agg == "count":
        agg_func = lambda lst: len(lst)
    elif agg == "sum":
        agg_func = lambda lst: sum(lst)
    elif agg == "mean":
        agg_func = lambda lst: sum(lst) / len(lst)
    else:
        agg_func = lambda lst: lst
        
    # aggregate
    inv_map = {key: agg_func(value) for key, value in inv_map.items()}
    
    # sort
    sorted_idx = np.array(list(inv_map.keys())).argsort()
    
    return np.array(list(inv_map.keys()))[sorted_idx], np.array(list(inv_map.values()))[sorted_idx]

def plot_graph_stats(graph):
    """
    Plot key statistics of a given networkx.graph instance
    (degree distribution, clustering coefficient, shortest path distribution)
    """
    fig, ax = plt.subplots(2, 2, figsize =(8, 6))
    fig.tight_layout(w_pad = 10, h_pad = 10)

    # degree distribution
    X, y = invert_dict_count(graph.degree(), agg = "count")
    ax[0,0].scatter(X, y, s = 10, marker = 'o', c = "w", linewidths = 0.5, edgecolors = "blue")
    ax[0,0].set_yscale("log")
    ax[0,0].set_xscale("log")
    ax[0,0].set_title("Degree Distribution")
    ax[0,0].set_xlabel("Degree (k)")
    ax[0,0].set_ylabel("Count (P(k) * n)")

    # clustering coefficient
    X, y = invert_dict_count(graph.degree(), y_dict = nx.clustering(graph), agg = "mean")
    ax[0,1].scatter(X, y, s = 10, marker = 'o', c = "w", linewidths = 0.5, edgecolors = "blue")
    ax[0,1].set_yscale("log")
    ax[0,1].set_xscale("log")
    ax[0,1].set_title("Clustering Coefficient")
    ax[0,1].set_xlabel("Degree (k)")
    ax[0,1].set_ylabel("Ø Clustering Coefficient")

    # shortest path
    ## compute
    inv_map = {}

    for idx, (node, path_length_dict) in enumerate(nx.shortest_path_length(graph)):    
        for k, v in path_length_dict.items():
            if v not in inv_map:
                inv_map[v] = 1
            else:
                inv_map[v] += 1

    path_length_mean = {k: v/(graph.number_of_nodes()**2) for k, v in inv_map.items()}

    ## sort
    sorted_idx = np.array(list(path_length_mean.keys())).argsort() # sort
    X = np.array(list(path_length_mean.keys()))[sorted_idx]
    y = np.array(list(path_length_mean.values()))[sorted_idx]

    ## plot distribution
    ax[1,0].scatter(X, y, s = 10, marker = 'o', c = "w", linewidths = 0.5, edgecolors = "blue")
    ax[1,0].set_title("Distribution of shortest path lengths")
    ax[1,0].set_xlabel("I (Path length in hops)")
    ax[1,0].set_yscale("log")
    ax[1,0].set_ylabel("P(I)")

    ## plot cumulative distribution
    ax[1,1].scatter(X, np.cumsum(y), s = 10, marker = 'o', c = "w", linewidths = 0.5, edgecolors = "blue")
    ax[1,1].set_title("Cum. Distribution of shortest path lengths")
    ax[1,1].set_xlabel("I (Path length in hops)")
    ax[1,1].set_ylabel("P(I)")

def plot_corr_matrix(df):
    """
    plot absolute correlation matrix of given pandas dataframe (all columns in df must be numerical!)
    """
    # plot correlation matrix
    corr_matrix = np.tril(np.abs(np.rint(np.array(df.corr()) * 100)), k = -1)

    # create labels
    labels = [f"{idx}: {col}" for idx, col in enumerate(df.columns)]

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(8,6))
    cm = ConfusionMatrixDisplay(corr_matrix,
                                display_labels = labels)
    cm.plot(ax = ax, xticks_rotation = 'vertical', cmap = plt.cm.Blues, text_kw = {"color": "w", "fontsize": 6})
    ax.set_xticklabels([i for i in range(len(df.columns))])
    ax.tick_params(axis='x', labelrotation = 0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Absolute correlation matrix of edge based features")

def plot_robustness_analysis(graph, node2values, k):
    '''
    :param graph:
    :param node2values: Dictionary of nodes with centrality values
    :param k: the number of nodes to be removed
    '''
    num_connected_components = []
    num_removed_edges = []
    gcc_sizes = []
    rest_sizes = []
    # Please write your code below    
    
    # Copy the given graph in order to keep it from modifications
    g = graph.copy()
    # Generate a node list with respect to the values given 
    # in the node2values dictionary in descending order
    pairs = sorted(node2values.items(), key=lambda x: x[1], reverse=True)
    node_list = [p[0] for p in pairs]
    # Get the first k nodes
    node_remove_list = node_list[:k]
    
    for node in node_remove_list:
        edges = g.degree(node) # nbr of edges of current node
        g.remove_node(node) # remove current node
        # log results
        num_connected_components.append(nx.number_connected_components(g))
        num_removed_edges.append(edges)
        gcc = g.subgraph(max(nx.connected_components(g), key=len))
        r = gcc.number_of_nodes() / float(g.number_of_nodes())
        gcc_sizes.append(r)
        rest_sizes.append(1-r)
    
    # plot
    plt.figure(figsize=(10,3))
    plt.subplot(1, 2, 1)
    ## number of connected components
    line1, = plt.plot(num_connected_components, label = "# connected components")
    line2, = plt.plot(np.cumsum(num_removed_edges), label = "# removed edges")
    plt.xlabel("# removed nodes (by descending centrality value)")
    plt.ylabel("# objects")
    plt.legend(handles=[line1, line2])
    ## size of GCC vs rest
    plt.subplot(1, 2, 2)
    line1, = plt.plot(gcc_sizes, label="GCC") 
    line2, = plt.plot(rest_sizes, label="Rest")
    plt.xlabel("# removed nodes (by descending centrality value)")
    plt.ylabel("% nodes")
    plt.legend(handles=[line1, line2])
    plt.subplots_adjust(wspace=0.5)
    plt.show()