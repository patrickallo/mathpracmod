"""
Module with helper-classes for AuthorNetwork
"""
# Imports
from bisect import insort
import logging
from itertools import combinations, permutations

import networkx as nx
import numpy as np
from pandas import DataFrame

import access_classes as ac


class AuthorFrame(DataFrame):
    """
    Subclass of DataFrame with authors as index
    """

    def __init__(self, author_color):
        super().__init__(
            {'color': list(author_color.values())},
            index=list(author_color.keys()))
        self.sort_index(inplace=True)
        self['word counts'] = np.zeros(self.shape[0])
        for i in range(1, 12):
            self["level {}".format(i)] = np.zeros(self.shape[0])


class AuthorInteractionGraph(nx.DiGraph):
    """
    Subclass of DiGraph with authors as nodes,
    and edges based on direct replies.
    """

    def __init__(self, author_names, thread_graph, no_loops):
        super().__init__()
        self.add_nodes_from(author_names)
        self.__i_graph_edges(thread_graph, no_loops)
        self = ac.scale_weights(self, "weight", "scaled_weight")

    def __i_graph_edges(self, thread_graph, no_loops):
        """Adds edges to interaction-graph."""
        for (source, dest) in thread_graph.edges():
            if no_loops and source == dest:
                continue
            source = thread_graph.node[source]
            source_author = source['com_author']
            source_time = source['com_timestamp']
            dest = thread_graph.node[dest]
            dest_author = dest['com_author']
            try:
                assert source_time >= dest['com_timestamp']
            except AssertionError:
                logging.warning("%s is not after %s", source, dest)
            try:
                edge = self[source_author][dest_author]
            except (AttributeError, KeyError):
                self.add_edge(
                    source_author, dest_author,
                    weight=1, timestamps=[source_time])
            else:
                edge['weight'] += 1
                insort(edge['timestamps'], source_time)
                assert edge['weight'] == len(edge['timestamps'])
        for _, _, data in self.edges(data=True):
            data['log_weight'] = np.log2(data['weight'])


class AuthorClusterGraph(nx.Graph):
    """
    Subclass of Graph with authors as nodes, and
    edges based on comments within episodes.
    """

    def __init__(self, author_names, author_episodes):
        super().__init__()
        self.add_nodes_from(author_names)
        self.__c_graph_edges(author_episodes)
        self = ac.scale_weights(self, "weight", "scaled_weight")

    def __c_graph_edges(self, author_episodes):
        """Adds edges to cluster-based graph"""
        for source_author, dest_author in combinations(
                author_episodes.keys(), 2):
            source_a_ep = {(thread, cluster): weight for
                           (thread, cluster, weight, _) in
                           list(author_episodes[source_author])}
            dest_a_ep = {(thread, cluster): weight for
                         (thread, cluster, weight, _) in
                         list(author_episodes[dest_author])}
            overlap = source_a_ep.keys() & dest_a_ep.keys()
            overlap = [(thread, cluster) for (thread, cluster) in list(
                overlap) if cluster is not np.nan]
            if overlap:
                source_w = [source_a_ep[key] for key in overlap]
                dest_w = [dest_a_ep[key] for key in overlap]
                weight = sum(source_w)
                assert weight == sum(dest_w)
                self.add_edge(source_author,
                              dest_author,
                              weight=weight,
                              log_weight=np.log2(weight),
                              simple_weight=len(overlap))


class AuthorClusterDiGraph(nx.DiGraph):
    """Subclass of DiGraph with authors as nodes, and edges
    based on comments within episodes"""

    def __init__(self, author_names, author_episodes):
        super().__init__()
        self.add_nodes_from(author_names)
        self.__c_graph_directed_edges(author_episodes)
        self = ac.scale_weights(self, "weight", "scaled_weight")

    def __c_graph_directed_edges(self, author_episodes):
        """Adds directed edges to cluster-based graph.
        Edge from a->b with weight w means that sum of all comments
        made by a in cluster in which b participated as well is w."""
        for source_author, dest_author in permutations(
                author_episodes.keys(), 2):
            source_a_ep = {(thread, cluster): a_weight for
                           (thread, cluster, _, a_weight) in
                           list(author_episodes[source_author])}
            dest_a_ep = [(thread, cluster) for
                         (thread, cluster, _, _) in
                         list(author_episodes[dest_author])]
            overlap = source_a_ep.keys() & dest_a_ep
            overlap = [(thread, cluster) for (thread, cluster) in list(
                overlap) if cluster is not np.nan]
            if overlap:
                source_w = [source_a_ep[key] for key in overlap]
                weight = sum(source_w)
                self.add_edge(source_author,
                              dest_author,
                              weight=weight)


class AuthorEpisodeBipartite(nx.Graph):
    """Subclass of Graph with authors and episodes as nodes,
    and author-episode affiliation as edges.
    Outlier-comments not part of any episodes do not show up because
    they are not in author_episodes.
    Edge from a->b with weight w means that a contributed w comments
    over all episodes to which b also contributed"""

    def __init__(self, author_names, author_episodes):
        super().__init__()
        self.add_nodes_from(author_names, bipartite=0)
        for author in author_episodes.keys():
            # list of (author, episode, weight) tuples with auth constant
            self.add_weighted_edges_from(
                [(author,
                  episode[:2],
                  episode[3]) for episode in author_episodes[author]])
