from math import ceil
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pandas import Series
import seaborn as sns

import access_classes as ac

from notebook_helper.access_funs import *


def comment_means(project, distr_data, n_rows=3):
    total_comments = distr_data.iloc[project - n_rows:project].sum()
    participated = (distr_data.iloc[project - n_rows:project] >= 1).sum()
    ratio = (total_comments / participated).dropna()
    return ratio


def atmost_n_at_i(project, distr_data, thresh, **args):
    ratio = comment_means(project, distr_data, **args)
    mask = (ratio <= thresh)
    return ratio[mask]


def unique_means(project, distr_data, **args):
    return np.unique(comment_means(project, distr_data, **args))


def projects_participated_after_i(project, distr_data, index, n_rows=3):
    return (distr_data.iloc[project:project + n_rows] >= 1).sum().loc[index]


def proportion_participated_after_i(project, distr_data, index, n_rows=3):
    participated = (distr_data.iloc[project:project + n_rows] >= 1).any(
        ).loc[index].sum()
    return participated / len(index)


def compute_diameters(networks):
    """Takes dictionary of networks, and computes diameter of the
    core-networks.
    Returns Series of data"""
    diameters = {}
    for key, network in networks.items():
        if key.endswith("core"):
            if network.is_directed():
                diameters[key] = (nx.diameter(network.to_undirected(as_view=True)))
            else:
                diameters[key] = (nx.diameter(network))
        else:
            diameters[key] = np.nan
    return Series(diameters)


def compute_reciprocity(networks):
    """Takes dictionary of networks, and computes reciprocity of each.
    If nx.reciprocity returns (), it replaces it with NaN.
    Returns Series of data"""
    reciprocity = {}
    for key, network in networks.items():
        rec = nx.reciprocity(network)
        reciprocity[key] = rec if rec > 0 else np.nan
    return Series(reciprocity)            


def compute_clustering(networks):
    """Takes dictionary of networks, and computes average clustering.
    Converts networks to undirected if needed.
    Returns Series of data."""
    clustering = {}
    for key, network in networks.items():
        if network.is_directed():
            clustering[key] = (nx.average_clustering(network.to_undirected(as_view=True)))
        else:
            clustering[key] = (nx.average_clustering(network))
    return Series(clustering)


def compute_global_metrics(networks):
    """Takes dictionary of networks, and computes global measures.
    Returns DataFrame."""
    N = Series({key: network.number_of_nodes() for key, network in
        networks.items()})
    diameters = compute_diameters(networks)
    reciprocity = compute_reciprocity(networks)
    transitivity = Series({key: nx.transitivity(network) for key, network in networks.items()})
    avg_clustering = compute_clustering(networks)
    all_metrics = pd.concat([N, diameters, reciprocity, transitivity, avg_clustering], axis=1)
    all_metrics.columns = ["N", "Diameter", "Reciprocity", "Transitivity", "Average Clustering"]
    all_metrics = all_metrics.reindex(["i_graph", "i_graph_core", "c_graph", "c_graph_core" , "c_dgraph", "c_dgraph_core"])
    all_metrics.index = pd.MultiIndex.from_product([["interaction", "cluster", "directed cluster"], ["All nodes", "Largest Component"]],
                             names=['Network', 'Subgraph'])
    return all_metrics


def compute_path_data(a_network):
    """Takes a networks, and computes path-lenths.
    Returns overview of distances."""
    path_data = Series(dict(nx.shortest_path_length(a_network))).apply(Series).apply(lambda x: x.value_counts()).T.fillna(0)
    return path_data


def compare_path_lengths(path_data, thresh=1):
    max_path = path_data.columns[-1]
    close = path_data[list(range(1, thresh+1))].sum(axis=1)
    far = path_data[list(range(thresh+1, int(max_path)+1))].sum(axis=1)
    return close, far


def central_given_path_lenghts(path_data, thresh=1):
    close, far = compare_path_lengths(path_data, thresh)
    central = path_data[close >= far]
    print("Central participants: ", len(central))
    return central


def peripheral_given_path_lenghts(path_data, thresh=1):
    close, far = compare_path_lengths(path_data, thresh)
    peripheral = path_data[close < far]
    print("Peripheral participants: ", len(peripheral))
    return peripheral


def plot_evolution_network(
        evol_data, top, graph_type="cluster", drop_first=True):
    assert graph_type in ["cluster", "interaction"]
    if drop_first:
        evol_data = evol_data.drop(evol_data.index[0])
    X_dims = [x.format(graph_type) for x in [
        "N* ({})", "ln(N*) ({})", "lnln(N*) ({})"]]
    [y.format(graph_type) for y in ["N* ({})", "ln(N*) ({})", "lnln(N*) ({})"]]
    _, ((axes1, axes2), (axes3, axes4)) = plt.subplots(2, 2, figsize=(20, 7))
    for X_dim, ax in zip(X_dims, [axes1, axes2, axes3]):
        sns.regplot(evol_data[X_dim], evol_data["avg shortest path-length ({})".format(graph_type)], ax=ax)
    for col in [t + " ({})".format(graph_type) for t in top]:
        sns.regplot(evol_data["N* ({})".format(graph_type)], evol_data[col], ax=axes4);
    end_size = evol_data[X_dims[0]].iloc[-1]
    axes4.plot([0, end_size], [0, end_size], "--", color="k", alpha=.5)
    axes4.set_ylim(0, end_size);
    axes4.set_xlim(evol_data["N* ({})".format(graph_type)].min()-1, evol_data["N* ({})".format(graph_type)].max()+1)
    axes4.set_ylabel("Degree of top-participants");


def draw_indiv_networks(
        pm_frame, project, final_network, networks=None, core=None,
        thread_type=None, show_labels=None, k=20, cols=4):
    if networks is None:
        print("No data supplied")
        return
    if not show_labels:
        show_labels = []
    project_data = pm_frame.loc[project]
    author_network = get_project_at(
        pm_frame, project, thread_type, stage=-1).network
    final_network = final_network.subgraph(core)
    pos = nx.spring_layout(final_network, k=k)
    num_plots = len(networks)
    cols = cols
    rows = ceil(num_plots / cols)
    _, axes = plt.subplots(rows, cols, figsize=(40, 10*rows))

    for ((thread, network), axes) in zip(networks.iteritems(), axes.flatten()):
        node_list = [node for node in core if node in network.nodes]
        color_list = author_network.author_frame["color"].loc[node_list].values
        size_list = [15 * project_data[thread_type, 'comment_counter'].loc[thread][node] for node in node_list]
        this_network = network.subgraph(node_list)
        # draw all nodes invisibly to ensure consistent spacing
        nx.draw_networkx_nodes(final_network, pos, node_color="k", node_size=0, alpha=.2, ax=axes)
        # draw visible parts
        nx.draw_networkx_nodes(this_network, pos, nodelist=node_list, node_color=color_list, node_size=size_list, cmap="tab20", ax=axes)
        nx.draw_networkx_edges(this_network.to_undirected(as_view=True), pos, alpha=.3, ax=axes)
        for participant in show_labels:
            if participant in node_list:
                nx.draw_networkx_labels(this_network, pos=pos, labels={participant: participant}, fontsize=5, ax=axes)
        axes.xaxis.set_ticks([])
        axes.yaxis.set_ticks([])
        axes.set_title("Thread {} (comments: {:.0f}, participants: {:.0f}, {} shown)".format(
            thread,
            project_data[thread_type, 'number of comments'].loc[thread],
            project_data[thread_type, 'number of authors'].loc[thread],
            len(node_list)))
        ac.fake_legend([1, 10, 50],
                       title="Number of comments",
                       fun=lambda x: x * 15,
                       alpha=.3, loc=4, ax=axes)