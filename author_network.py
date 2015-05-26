"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a comment_thread.
"""

import networkx as nx
from comment_thread import *


## find way to deal with "anonymous"

a_thread = CommentThreadPolymath('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')

the_authors = [data['com_author'] for (node_id, data) in a_thread.graph.nodes_iter(data=True) if data['com_type'] == 'comment']
the_authorset = set(the_authors)
author_graph = nx.DiGraph()
author_graph.add_nodes_from(the_authorset) ## need to filter out pingbacks
for (source, dest) in a_thread.graph.edges_iter():
    source, dest = a_thread.graph.node[source]['com_author'], a_thread.graph.node[dest]['com_author']
    if not (source, dest) in author_graph.edges():
        author_graph.add_weighted_edges_from([(source, dest, 1)])
    else:
        author_graph[source][dest]['weight'] += 1


node_color = {author_node : a_thread.author_color[author_node] for author_node in the_authors}

nx.draw_networkx(author_graph,
                 node_size=500,
                 nodelist=node_color.keys(),
                 node_color=node_color.values())
plt.show()