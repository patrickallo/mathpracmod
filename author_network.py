"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a comment_thread.
"""

import networkx as nx
import matplotlib.pyplot as plt
from comment_thread import CommentThreadPolymath


## find way to deal with "anonymous"

THIS_THREAD = CommentThreadPolymath('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')

class AuthorNetwork(object):
    """Creates and draws Weighted nx.DiGraph of comments between authors."""
    def __init__(self, a_thread):
        self.a_thread = a_thread
        self.the_authors = [data['com_author'] for (node_id, data) in
                            self.a_thread.graph.nodes_iter(data=True) if
                            data['com_type'] == 'comment']
        self.the_authorset = set(self.the_authors)
        self.author_graph = nx.DiGraph()
        self.author_graph.add_nodes_from(self.the_authorset)
        for (source, dest) in self.a_thread.graph.edges_iter():
            source = self.a_thread.graph.node[source]['com_author']
            dest = self.a_thread.graph.node[dest]['com_author']
            if not (source, dest) in self.author_graph.edges():
                self.author_graph.add_weighted_edges_from([(source, dest, 1)])
            else:
                self.author_graph[source][dest]['weight'] += 1

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
                         node_size=500,
                         nodelist=node_color.keys(),
                         node_color=node_color.values())
        plt.show()

THIS_GRAPH = AuthorNetwork(THIS_THREAD)
THIS_GRAPH.draw_graph()