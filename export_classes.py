"""
Module for mixin_classes common to author_network, comment_thread,
and multi_comment_thread.
It uses: networkx
"""

import networkx as nx

class GraphExportMixin(object):
    """exports an nx.graph for external use"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def to_gephi(self):
        """Exports the full graph to gephi"""
        file_name = raw_input("Save as: ")
        nx.write_gexf(self.graph,
                      file_name+".gexf",
                      encoding='utf-8',
                      prettyprint=True,
                      version='1.2draft')

    def to_yaml(self):
        "Exports the full graph to yaml"
        file_name = raw_input("Save as: ")
        nx.write_yaml(self.graph,
                      file_name+".yaml",
                      encoding='utf-8')
