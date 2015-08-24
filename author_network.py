"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a multi_comment_thread.
"""
# Imports
from collections import Counter
import sys
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from comment_thread import MultiCommentThread, CommentThreadPolymath, CommentThreadGowers, CommentThreadTerrytao
import export_classes as ec

# Loading settings
with open("settings.yaml", "r") as settings_file:
    SETTINGS = yaml.safe_load(settings_file.read())

# Main
def main(urls, thread_type="Polymath"):
    """Creates AuthorNetwork based on supplied list of urls, and draws graph."""
    # TODO: fix mismatch between many urls and one type
    try:
        the_threads = eval("[CommentThread{}(url) for url in {}]".format(thread_type, urls))
    except ValueError as err:
        print err
        the_threads = []
    an_mthread = MultiCommentThread(*the_threads)
    a_network = AuthorNetwork(an_mthread)
    show_or_return = raw_input("Show graph or return object? (graph / object) ")
    if show_or_return.lower() == "graph":
        a_network.draw_graph()
    elif show_or_return.lower() == "object":
        return a_network
    else:
        raise ValueError("Invalid choice.")

# Classes
class AuthorNetwork(ec.GraphExportMixin, object):
    """
    Creates and draws Weighted nx.DiGraph of comments between authors.

    Attributes:
        mthread: MultiCommentThread object.
        all_thread_graphs: DiGraph of MultiCommentThread.
        node_name: dict with nodes as keys and authors as values.
        author_color: dict with authors as keys and colors (ints) as values.
        graph: weighted nx.DiGraph (weighted edges between authors)

    Methods:
        author_count: returns Counter-object (dict) with
                      authors as keys and num of comments as values
        plot_author_count: plots author_count
        author_report: returns dict with comments,
                       replies and direct replies,
                       and comments per level for a given author
        weakly connected components: returns generator of weakly connected components
        draw_graph: draws author_network

    """
    def __init__(self, an_mthread):
        super(AuthorNetwork, self).__init__()
        self.mthread = an_mthread
        self.all_thread_graphs = an_mthread.graph
        self.node_name = an_mthread.node_name
        self.author_color = an_mthread.author_color
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.author_color.keys())
        for (source, dest) in self.all_thread_graphs.edges_iter():
            source = self.all_thread_graphs.node[source]['com_author']
            dest = self.all_thread_graphs.node[dest]['com_author']
            if not (source, dest) in self.graph.edges():
                self.graph.add_weighted_edges_from([(source, dest, 1)])
            else:
                self.graph[source][dest]['weight'] += 1

    def author_count(self):
        """Returns dict with count of authors"""
        return Counter(self.node_name.values())

    def plot_author_count(self):
        """Shows plot of author_count"""
        labels, values = zip(*self.author_count().items())
        indexes = np.arange(len(labels))
        plt.bar(indexes, values, 1)
        plt.xticks(indexes + 0.5, labels, rotation='vertical')
        plt.show()

    def author_report(self, an_author):
        """Returns author-report as dict"""
        # list of (node_id, depth) tuple of type(string, int)
        the_comments_levels = [(node_id, data["com_depth"]) for (node_id, data) in
                               self.all_thread_graphs.nodes_iter(data=True) if
                               data["com_author"] == an_author]
        # list, list
        the_comments, the_levels = zip(*the_comments_levels)
        return {
            "number of comments made" : len(the_comments),
            "comments by level" : Counter(the_levels),
            "direct replies" : sum((self.mthread.comment_report(i)["direct replies"]
                                    for i in the_comments)),
            "indirect replies (all, pure)" : tuple((sum(lst) for lst in
                                                    zip(*(self.mthread.comment_report(i)
                                                          ["indirect replies (all, pure)"]
                                                          for i in the_comments))))
            }

    def w_connected_components(self):
        """Returns weakly connected components as generator of list of nodes.
        This ignores the direction of edges."""
        return nx.weakly_connected_components(self.graph)

    def draw_graph(self):
        """Draws and shows graph."""
        show_labels = raw_input("Show labels? (default = yes) ")
        show_labels = show_labels.lower() != 'no'
        # attributing widths to edges
        edges = self.graph.edges()
        weights = [self.graph[source][dest]['weight'] / float(10) for source, dest in edges]
        # positions with spring
        positions = nx.spring_layout(self.graph, k=.7, scale=2)
        # actual drawing
        nx.draw_networkx(self.graph, positions,
                         with_labels=show_labels,
                         font_size=7,
                         node_size=1000,
                         nodelist=self.author_color.keys(),
                         node_color=self.author_color.values(),
                         edges=edges,
                         width=weights)
        plt.show()


if __name__ == '__main__':
    ARGUMENTS = sys.argv[1:]
    if ARGUMENTS:
        main(ARGUMENTS)
    else:
        print SETTINGS['msg']
        main([SETTINGS['url']], SETTINGS['type'])
