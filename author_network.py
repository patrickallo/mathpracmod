"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a comment_thread.
"""

import sys
from collections import Counter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from comment_thread import CommentThreadPolymath


def main(url):
    """Creates AuthorNetwork based on supplied url, and draws graph."""
    a_thread = CommentThreadPolymath(url)
    a_graph = AuthorNetwork(a_thread)
    show_or_return = raw_input("Show graph or return object? (graph / object) ")
    if show_or_return.lower() == "graph":
        a_graph.draw_graph()
    elif show_or_return.lower() == "object":
        return a_graph
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
        the_authors: list of all authors (repetitions removed at init)
        author_graph: weighted nx.DiGraph
    """
    def __init__(self, a_thread):
        self.a_thread = a_thread
        self.the_authors = (data['com_author'] for (node_id, data) in
                            self.a_thread.graph.nodes_iter(data=True) if
                            data['com_type'] == 'comment')
        self.the_authors = list(set(self.the_authors))
        self.author_graph = nx.DiGraph()
        self.author_graph.add_nodes_from(self.the_authors)
        for (source, dest) in self.a_thread.graph.edges_iter():
            source = self.a_thread.graph.node[source]['com_author']
            dest = self.a_thread.graph.node[dest]['com_author']
            if not (source, dest) in self.author_graph.edges():
                self.author_graph.add_weighted_edges_from([(source, dest, 1)])
            else:
                self.author_graph[source][dest]['weight'] += 1

    def author_count(self):
        """returns dict with count of authors"""
        return Counter(nx.get_node_attributes(self.a_thread.graph, "com_author").values())

    def plot_author_count(self):
        """shows plot of author_count"""
        labels, values = zip(*self.author_count().items())
        indexes = np.arange(len(labels))
        plt.bar(indexes, values, 1)
        plt.xticks(indexes + 0.5, labels, rotation='vertical')
        plt.show()

    def draw_graph(self):
        """Draws and shows graph."""
        show_labels = raw_input("Show labels? (default = yes) ")
        show_labels = show_labels.lower() != 'no'
        # attributing colors
        node_color = {author_node : self.a_thread.author_color[author_node]
                      for author_node in self.the_authors}
        # actual drawing
        nx.draw_networkx(self.author_graph,
                         with_labels=show_labels,
                         font_size=8,
                         node_size=200,
                         nodelist=node_color.keys(),
                         node_color=node_color.values())
        plt.show()


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError:
        print "testing with Minipolymath 4"
        main('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')
