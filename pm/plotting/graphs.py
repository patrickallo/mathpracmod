"""
Module with stand-alone graph-plotting functions
"""

# Imports
from math import log
from matplotlib.dates import date2num, DateFormatter, DayLocator
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite

import access_classes as ac
from author_network import AuthorNetwork
from multi_comment_thread import MultiCommentThread


SETTINGS, CMAP, _ = ac.load_settings()


# Functions
def draw_discussion_tree(mthread_or_graph, **kwargs):
    """Plots discussion_tree
    Should be called with project as kwarg for correct title."""
    intervals = kwargs.pop("intervals", 10)
    first = kwargs.pop("first", SETTINGS['first_date'])
    last = kwargs.pop("last", SETTINGS['last_date'])
    remove_title = kwargs.pop("remove_title", False)
    project, show, _ = ac.handle_kwargs(**kwargs)
    if isinstance(mthread_or_graph, MultiCommentThread):
        discussion_tree = mthread_or_graph.graph
        blog_nodes = mthread_or_graph.type_nodes
        author_color = mthread_or_graph.author_color
        node_name = mthread_or_graph.node_name
    elif isinstance(mthread_or_graph, nx.classes.digraph.DiGraph):
        discussion_tree = mthread_or_graph
        try:
            blog_nodes = kwargs.pop("blog_nodes")
            author_color = kwargs.pop("author_color")
            node_name = kwargs.pop("node_name")
        except KeyError:
            raise ValueError
    else:
        raise ValueError
    # creating title and axes
    figure = plt.figure()
    if not remove_title:
        figure.suptitle("Thread structure for {}".format(project).title(),
                        fontsize=12)
    axes = figure.add_subplot(111)
    axes.yaxis.set_major_locator(DayLocator(interval=intervals))
    axes.yaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks(list(range(1, 11)))
    axes.set_xlabel("Comment Levels")
    axes.xaxis.set_ticks_position('bottom')
    first, last, *_ = ac.check_date_type(first, last)
    dates = sorted([data["com_timestamp"] for _, data in
                    discussion_tree.nodes(data=True)])
    first, last = max(first, dates[0]), min(last, dates[-1])
    plt.ylim(first, last)
    # creating and drawingsub_graphs
    types_markers = {thread_type: marker for (thread_type, marker) in
                     zip(blog_nodes.keys(),
                         ['o', '>', 'H', 'D'][:len(
                             blog_nodes.keys())])}
    for (thread_type, marker) in types_markers.items():
        type_subgraph = discussion_tree.subgraph(blog_nodes[thread_type])
        # generating colours and positions for sub_graph
        positions = {node_id: (data["com_depth"],
                               date2num(data["com_timestamp"]))
                     for (node_id, data) in
                     type_subgraph.nodes(data=True)}
        node_color = {node_id: (author_color[node_name[node_id]])
                      for node_id in type_subgraph.nodes()}
        # drawing nodes of type_subgraph
        nx.draw_networkx_nodes(type_subgraph, positions,
                               node_size=200,
                               nodelist=list(node_color.keys()),
                               node_color=list(node_color.values()),
                               node_shape=marker,
                               vmin=SETTINGS['vmin'],
                               vmax=SETTINGS['vmax'],
                               cmap=CMAP,
                               ax=axes)
        nx.draw_networkx_edges(type_subgraph, positions, width=.5)
        # nx.draw_networkx_labels(type_subgraph, positions, fontsize=4)
        if SETTINGS['show_labels_comments']:
            nx.draw_networkx_labels(
                type_subgraph, positions, font_size=8,
                labels={node: node[9:] for node in node_color.keys()})
    # show all
    plt.style.use(SETTINGS['style'])
    the_lines = [mlines.Line2D([], [], color='gray',
                               marker=marker,
                               markersize=5,
                               label=thread_type[13:])
                 for (thread_type, marker) in types_markers.items()]
    plt.legend(title="Where is the discussion happening",
               handles=the_lines)
    ac.show_or_save(show)


def draw_discussion_tree_radial(mthread_or_graph, **kwargs):
    """Plots discussion_tree
    Should be called with project as kwarg for correct title."""
    project, show, _ = ac.handle_kwargs(**kwargs)
    if isinstance(mthread_or_graph, MultiCommentThread):
        discussion_tree = mthread_or_graph.graph
        author_color = mthread_or_graph.author_color
    elif isinstance(mthread_or_graph, nx.classes.digraph.DiGraph):
        discussion_tree = mthread_or_graph
        try:
            author_color = kwargs.pop("author_color")
        except KeyError:
            raise ValueError
    else:
        raise ValueError
    discussion_tree = discussion_tree.reverse(copy=True)
    discussion_tree.add_node("root", com_depth=0)
    for node in discussion_tree.nodes(data=True):
        if node[1]['com_depth'] == 1:
            discussion_tree.add_edge("root", node[0])
    node_color = {
        int(node_id[8:]): author_color[discussion_tree.node[
            node_id]['com_author']] for node_id in discussion_tree.nodes() if
        node_id != 'root'}
    # node_color[0] = 0
    mapping = {node: node[8:] for node in discussion_tree.nodes()}
    mapping["root"] = 0
    nx.relabel_nodes(discussion_tree, lambda x: int(mapping[x]), copy=False)
    discussion_tree = nx.Graph(discussion_tree.edges())
    tree_pos = nx.nx_pydot.graphviz_layout(discussion_tree,
                                           prog='twopi', root=0, args='')
    _, axes = plt.subplots()
    axes.xaxis.set_ticks([])
    axes.yaxis.set_ticks([])
    axes.set_title("Thread structure for {}".format(project).title())
    nx.draw_networkx_edges(discussion_tree, tree_pos, width=.5, alpha=.3)
    # nx.draw_networkx_labels(discussion_tree, tree_pos, fontsize=4)
    nx.draw_networkx_nodes(discussion_tree.subgraph(0), {0: tree_pos.pop(0)},
                           node_size=150,
                           node_list=[0],
                           node_color='k',
                           ax=axes)
    nx.draw_networkx_nodes(discussion_tree.subgraph(node_color.keys()),
                           tree_pos,
                           node_size=150,
                           nodelist=list(node_color.keys()),
                           node_color=list(node_color.values()),
                           vmin=SETTINGS['vmin'],
                           vmax=SETTINGS['vmax'],
                           cmap=CMAP, ax=axes)
    plt.style.use(SETTINGS['style'])
    ac.show_or_save(show)


def draw_author_network(network_or_graph, **kwargs):
    """Draws and show author_network graph"""
    # TODO: consider adding thresh-value for drawing edges!
    # especially for cluster-based networks
    k = kwargs.pop("k", None)
    reset = kwargs.pop("reset", False)
    project, show, fontsize = ac.handle_kwargs(**kwargs)
    weight = kwargs.pop("weight", "weight")
    remove_title = kwargs.pop("remove_title", False)
    if isinstance(network_or_graph, AuthorNetwork):
        graph_type = kwargs.pop("graph_type", "interaction")
        if graph_type == "cluster":
            graph = network_or_graph.c_graph
            graph_type = "Co-location Network"
        elif graph_type == "directed cluster":
            graph = network_or_graph.c_dgraph
            graph_type = "Directed Co-location Network"
        elif graph_type == "interaction":
            graph = network_or_graph.i_graph
            graph_type = "Interaction Network"
    else:
        raise NotImplementedError
    # attributing widths and colors to edges
    edges = graph.edges()
    weights = [graph[source][dest][weight] * 15 for
               source, dest in edges]
    edge_colors = [plt.cm.Blues(weight) for weight in weights]
    # attributes sizes to nodes
    sizes = [(log(network_or_graph.author_count()[author], 2) + 1) * 300
             for author in network_or_graph.author_frame.index]
    # positions with spring
    if reset or not network_or_graph.positions:
        network_or_graph.positions = nx.spring_layout(
            graph, k=k, scale=1)
    # creating title and axes
    figure = plt.figure()
    if not remove_title:
        figure.suptitle("{} for {}".format(graph_type, project).title(),
                        fontsize=12)
    axes = figure.add_subplot(111)
    axes.xaxis.set_ticks([])
    axes.yaxis.set_ticks([])
    # actual drawing
    plt.style.use(SETTINGS['style'])
    nx.draw_networkx(graph, network_or_graph.positions,
                     with_labels=SETTINGS['show_labels_authors'],
                     font_size=fontsize,
                     node_size=sizes,
                     nodelist=network_or_graph.author_frame.index.tolist(),
                     node_color=network_or_graph.author_frame[
                         'color'].tolist(),
                     edges=edges,
                     width=1,
                     edge_color=edge_colors,
                     vmin=SETTINGS['vmin'],
                     vmax=SETTINGS['vmax'],
                     cmap=CMAP,
                     ax=axes)
    ac.fake_legend([1, 5, 20],
                   title="Number of comments",
                   fun=lambda x: (log(x, 2) + 1) * 300,
                   alpha=.3)
    ac.show_or_save(show)


def draw_bipartite_network(network_or_graph, **kwargs):
    """Draws bipartitite author-episode affiliation-network"""
    # consider to expand this to author-project and author-thread affils
    project, show, fontsize = ac.handle_kwargs(**kwargs)
    remove_title = kwargs.pop("remove_title", False)
    if isinstance(network_or_graph, AuthorNetwork):
        graph = network_or_graph.bp_graph
    elif bipartite.is_bipartite(graph):
        graph = network_or_graph
    else:
        raise TypeError("Need AuthorNetwork of Bipartite Graph.")
    auths = {n for n, d in graph.nodes(data=True) if 0 in d.values()}
    episodes = set(graph) - auths
    scale = len(auths) // len(episodes)
    a_pos = {n: (1, i) for i, n in enumerate(auths)}
    a_colors = [network_or_graph.author_frame.loc[author, 'color']
                for author in a_pos.keys()]
    e_sizes = set.union(
        *network_or_graph.author_frame.episodes.dropna().tolist())
    e_sizes = {(thr, clus): weight for thr, clus, weight, _ in e_sizes}
    e_pos = {n: (2, i * scale * 2) for i, n in enumerate(episodes)}
    edge_pos = a_pos.copy()
    edge_pos.update(e_pos)
    # creating title and axes
    figure = plt.figure()
    if not remove_title:
        figure.suptitle("Affiliation-network for {}".format(project).title(),
                        fontsize=12)
    axes = figure.add_subplot(111)
    axes.xaxis.set_ticks([])
    axes.yaxis.set_ticks([])
    # actual drawing
    plt.style.use(SETTINGS['style'])
    nx.draw_networkx_nodes(graph,
                           pos=a_pos, nodelist=a_pos.keys(),
                           node_color=a_colors,
                           node_size=25,
                           ax=axes, cmap=CMAP)
    nx.draw_networkx_nodes(graph,
                           pos=e_pos, nodelist=e_pos.keys(),
                           node_size=[e_sizes[ep] * 10
                                      for ep in e_pos.keys()],
                           node_color="gray", alpha=.3,
                           ax=axes)
    nx.draw_networkx_edges(graph,
                           pos=edge_pos,
                           alpha=.5,
                           ax=axes)
    ac.fake_legend([10, 50, 100],
                   title="Number of comments",
                   fun=lambda x: x * 10,
                   alpha=.3)
    ac.show_or_save(show)
