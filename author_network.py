"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a comment_thread.
"""

import networkx as nx
from comment_thread import *


## check out alternative via nx.from_numpy_matrix(A,create_using=nx.DiGraph())

## should re-use colors from original comment_thread
## find way to deal with "anonymous"

a_thread = CommentThreadPolymath('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')

the_authors = set(nx.get_node_attributes(a_thread.graph, "com_author").values())
author_graph = nx.DiGraph()
author_graph.add_nodes_from(the_authors) ## need to filter out pingbacks
for (source, dest) in a_thread.graph.edges_iter():
    print source, dest
    source, dest = a_thread.graph.node[source]['com_author'], a_thread.graph.node[dest]['com_author']
    print source, dest
    if not (source, dest) in author_graph.edges():
        print "add new edge"
        author_graph.add_weighted_edges_from([(source, dest, 1)])
    else:
        print "inc edge"
        author_graph[source][dest]['weight'] += 1

nx.draw_networkx(author_graph)
plt.show()