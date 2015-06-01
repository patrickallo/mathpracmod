"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a comment_thread.
"""

import sys
from collections import Counter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from comment_thread import MultiCommentThread, CommentThreadPolymath


def main(url):
    """Creates AuthorNetwork based on supplied url, and draws graph."""
    an_mthread = MultiCommentThread(CommentThreadPolymath(url, comments_only=True))
    a_network = AuthorNetwork(an_mthread)
    show_or_return = raw_input("Show graph or return object? (graph / object) ")
    if show_or_return.lower() == "graph":
        a_network.draw_graph()
    elif show_or_return.lower() == "object":
        return a_network
    else:
        raise ValueError("Invalid choice.")

class AuthorNetwork(object):
    """Creates and draws Weighted nx.DiGraph of comments between authors.

    Methods:
        author_count: returns Counter-object (dict) with
                      authors as keys and num of comments as values
        plot_author_count: plots author_count
        draw_graph: draws author_network

    Attributes:
        a_thread: Comment_Thread object
        the_authors: list of all authors
        author_graph: weighted nx.DiGraph
    """
    def __init__(self, an_mthread):
        self.all_thread_graphs = an_mthread.threads_graph
        self.node_name = an_mthread.node_name
        self.author_color = an_mthread.author_color
        self.author_graph = nx.DiGraph()
        self.author_graph.add_nodes_from(self.author_color.keys())
        for (source, dest) in self.all_thread_graphs.edges_iter():
            source = self.all_thread_graphs.node[source]['com_author']
            dest = self.all_thread_graphs.node[dest]['com_author']
            if not (source, dest) in self.author_graph.edges():
                self.author_graph.add_weighted_edges_from([(source, dest, 1)])
            else:
                self.author_graph[source][dest]['weight'] += 1

    def author_count(self):
        """returns dict with count of authors"""
        return Counter(self.node_name)

    def plot_author_count(self):
        """shows plot of author_count"""
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
        print the_comments
        return {
            "number of comments" : len(the_comments),
            "comments by level" : Counter(the_levels),
            "direct replies" : sum((self.all_thread_graphs.comment_report(i)["direct replies"]
                                    for i in the_comments)),
            "indirect replies (all, pure)" : tuple((sum(lst) for lst in
                                                    zip(*(self.all_thread_graphs.comment_report(i)
                                                          ["indirect replies (all, pure)"]
                                                          for i in the_comments))))
            }

    def w_connected_components(self):
        """Returns weakly connected components as generator of list of nodes.
        This ignores the direction of edges."""
        return nx.weakly_connected_components(self.author_graph)

    def draw_graph(self):
        """Draws and shows graph."""
        show_labels = raw_input("Show labels? (default = yes) ")
        show_labels = show_labels.lower() != 'no'
        # attributing widths to edges
        edges = self.author_graph.edges()
        weights = [self.author_graph[source][dest]['weight'] / float(10) for source, dest in edges]
        # positions with spring
        positions = nx.spring_layout(self.author_graph, k=.7, scale=2)
        # actual drawing
        nx.draw_networkx(self.author_graph, positions,
                         with_labels=show_labels,
                         font_size=7,
                         node_size=1000,
                         nodelist=self.author_color.keys(),
                         node_color=self.author_color.values(),
                         edges=edges,
                         width=weights)
        plt.show()


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError:
        print "testing with Minipolymath 4"
        main('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')
