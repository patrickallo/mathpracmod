"""
Module for mixin_classes common to author_network and comment_thread.
It uses: networkx.
"""

import networkx as nx

class GraphExport(object):
    """exports an nx.graph to gephi"""
    @classmethod
    def to_gephi(cls, a_graph):
        """Exports the full graph to gephi"""
        file_name = raw_input("Save as: ")
        nx.write_gexf(a_graph,
                      file_name+".gexf",
                      encoding='utf-8',
                      prettyprint=True,
                      version='1.2draft')

    @classmethod
    def to_yaml(cls, a_graph):
        "Exports the full graph to yaml"
        file_name = raw_input("Save as: ")
        nx.write_yaml(a_graph,
                      file_name+".yaml",
                      encoding='utf-8')
