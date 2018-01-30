"""Plotting-functions to analyse connected components and network-metrics
when edges and/or nodes are removed"""
# imports
import matplotlib.pyplot as plt
import networkx as nx
from numpy import unique
from pandas import DataFrame

from author_network import SETTINGS


def remove_edges_of_weigth_k(graph, k, weight='weight'):
    """Removed edges of graph with weight equal to k, and
    returns resulting graph"""
    out = graph.copy()
    out.remove_edges_from((edge for (edge, weight) in nx.get_edge_attributes(
        graph, weight).items() if weight == k))
    return out


def init_component_array(graph, components):
    """Returns 0-padded list of list of components of graph."""
    first_row = sorted([len(x) for x in components(graph)], reverse=True)
    row_len = sum(first_row)
    pad = (row_len - len(first_row)) * [0]
    return [first_row + pad], row_len


def add_component_row(graph, components, in_data, row_len):
    """Appends 0-padded list of components to list of lists of components"""
    new_row = sorted([len(x) for x in components(graph)], reverse=True)
    pad = (row_len - len(new_row)) * [0]
    in_data.append(new_row + pad)


def plot_component_data(data, network_data, top_title):
    _, axes = plt.subplots(2, 1)
    data.plot(kind="bar", stacked=True, ax=axes[0], cmap=SETTINGS['cmap'])
    network_data.plot(kind="line", ax=axes[1])
    axes[0].legend_.remove()
    axes[0].xaxis.set_visible(False)
    y_values = unique(data.iloc[:, 0].values)
    axes[0].yaxis.set_ticks(y_values)
    axes[0].yaxis.set_ticklabels(y_values, fontsize=10)
    axes[0].set_title(top_title)
    axes[1].legend(loc=1)
    axes[1].xaxis.set_ticks(data.index)
    axes[1].xaxis.set_ticklabels(data.index)
    axes[1].set_xlim(data.index[0] - .5, data.index[-1] + .5)
    axes[1].set_title("Network measures")


def components_setup(author_network, g_type, project=None):
    if g_type == "interaction":
        graph = author_network.i_graph.copy()
        components = nx.weakly_connected_components
        top_title = "Weakly connected components in {} ({})".format(
            project, g_type)
    elif g_type == "cluster":
        graph = author_network.c_graph.copy()
        components = nx.connected_components
        top_title = "Connected components in {} ({})".format(
            project, g_type)
    elif g_type == "directed cluster":
        graph = author_network.c_dgraph.copy()
        components = nx.weakly_connected_components
        top_title = "Weakly connected components in {} ({})".format(
            project, g_type)
    else:
        raise TypeError
    return graph, components, top_title


def shrinking_components_nodes(author_network, g_type, n,
                               measure="degree centrality", weight="weight",
                               project=None, print_removed=False):
    """Creates DataFrame of components based on removal of most connected
    nodes of an author_network, and plots bar and line-plot."""
    graph, components, top_title = components_setup(
        author_network, g_type, project)
    sorted_nodes = author_network._AuthorNetwork__get_centrality_measures(
        g_type=g_type, measures=[measure], weight=weight)
    for_df, row_len = init_component_array(graph, components)
    transitivity = [nx.transitivity(graph)]
    clustering = [nx.average_clustering(graph.to_undirected(), weight=None)]
    clustering_w = [nx.average_clustering(
        graph.to_undirected(), weight="weight")]
    for k in range(1, n + 1):
        to_remove = list(sorted_nodes[:k].index)
        graph.remove_nodes_from(to_remove)
        add_component_row(graph, components, for_df, row_len)
        transitivity.append(nx.transitivity(graph))
        clustering.append(nx.average_clustering(
            graph.to_undirected(), weight=weight))
        clustering_w.append(nx.average_clustering(
            graph.to_undirected(), weight=weight))
    data = DataFrame(for_df)
    data.columns.name = "Number of participants"
    data.index.name = "Number of most connected nodes removed"
    network_data = DataFrame({"transitivity": transitivity,
                              "average clustering": clustering,
                              "average clustering (weighted)": clustering_w},
                             index=data.index)
    plot_component_data(data, network_data, top_title)
    if print_removed:
        print(sorted_nodes.iloc[:n])


def shrinking_components_edges(author_network, g_type, n, weight="weight",
                               project=None):
    graph, components, top_title = components_setup(
        author_network, g_type, project)
    for_df, row_len = init_component_array(graph, components)
    transitivity = [nx.transitivity(graph)]
    clustering = [nx.average_clustering(graph.to_undirected(), weight=None)]
    clustering_w = [nx.average_clustering(
        graph.to_undirected(), weight=weight)]
    for k in range(1, n + 1):
        graph = remove_edges_of_weigth_k(graph, k, weight=weight)
        add_component_row(graph, components, for_df, row_len)
        transitivity.append(nx.transitivity(graph))
        clustering.append(nx.average_clustering(
            graph.to_undirected(), weight=None))
        clustering_w.append(nx.average_clustering(
            graph.to_undirected(), weight="weight"))
    data = DataFrame(for_df)
    data.columns.name = "Number of participants"
    data.index = data.index + 1
    data.index.name = "Threshold for edge-weight"
    network_data = DataFrame({"transitivity": transitivity,
                              "average clustering": clustering,
                              "average clustering (weighted)": clustering_w},
                             index=data.index)
    plot_component_data(data, network_data, top_title)
