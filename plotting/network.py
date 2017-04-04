"""module with wrapper-functions to access plotting-functions from
AuthorNetwork"""
# imports
from functools import partial
from operator import methodcaller
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
import access_classes as ac
from notebook_helper.access_funs import get_project_at, thread_or_project
from plotting.graphs import draw_author_network
from plotting.components import (shrinking_components_edges,
                                 shrinking_components_nodes)

# Loading settings
SETTINGS, CMAP = ac.load_settings()


def plot_from_network(plot_method, pm_frame, project, **kwargs):
    """Auxiliary-function calling supplied plot_method of AuthorNetwork
    from project"""
    data = get_project_at(pm_frame, project, kwargs["thread_type"],
                          kwargs["stage"])["network"]
    methodcaller(plot_method, project=project, **kwargs)(data)


def plot_from_network_tp(plot_method, pm_frame, project, **kwargs):
    """Auxiliary-function calling supplied plot_method of AuthorNetwork
    from thread or project."""
    project, data = thread_or_project(pm_frame, project, **kwargs)
    methodcaller(plot_method, project=project, **kwargs)(data)


def plot_activity_area(pm_frame, project, thread_type):
    """Plots stacked area-plot of accumulated number of comments for each
    participant per thread."""
    if thread_type == "research threads":
        mask = pm_frame["basic", "research"].loc[project]
    elif thread_type == "discussion threads":
        mask = ~pm_frame["basic", "research"].loc[project]
    else:
        mask = np.ones_like(pm_frame["basic", "research"].loc[project],
                            dtype=bool)
    data = pm_frame[
        thread_type, "comment_counter (accumulated)"].loc[project].apply(
            Series).dropna(
                axis=1, how='all')[mask]
    data.sort_values(axis=1, by=data.index[-1], inplace=True)
    data.index.name = "Threads"
    author_color = get_project_at(
        pm_frame, project, thread_type, stage=-1).loc[
            'mthread (accumulated)'].author_color
    colors = ac.color_list([author_color[author] for author in data.columns],
                           SETTINGS['vmin'], SETTINGS['vmax'], cmap=CMAP)
    _, axes = plt.subplots()
    axes.xaxis.set_ticks(range(data.index[-1] + 1))
    data.plot(kind="area", stacked=True, ax=axes,
              title="Growth of number of comments per author in {}".format(
                  project), legend=None, color=colors, fontsize=8)


plot_author_activity_bar = partial(
    plot_from_network, 'plot_author_activity_bar',
    thread_type="all threads", stage=-1, what="by level", fontsize=6)
plot_author_activity_bar.__doc__ = """Plots bar-chart of comment_activity of
    commenters in project. 'what' can be 'by level' or 'word counts'"""

plot_centrality_measures = partial(
    plot_from_network, "plot_centrality_measures",
    thread_type="all threads", stage=-1,
    graph_type='interaction',
    delete_on=None, thresh=0,
    fontsize=6)
plot_centrality_measures.__doc__ = """
    Plots line-chart of different centrality-measures."""

plot_activity_degree = partial(
    plot_from_network, "plot_activity_degree",
    thread_type="all threads", stage=-1,
    graph_type='interaction', weight=None,
    measures=None, delete_on=None, thresh=0,
    fontsize=6)
plot_activity_degree.__doc__ = """
    Plots superposition of bar-chart of comment-activity and
    line-chart of degree-centrality"""

plot_activity_prop = partial(
    plot_from_network, "plot_activity_prop",
    thread_type="all threads", stage=-1)
plot_activity_prop.__doc__ = """Plot of number of comments (bar) and
    proportion level-1 / higher-level comment (line) for all authors"""

plot_activity_pie = partial(
    plot_from_network, "plot_author_activity_pie",
    thread_type="all threads", stage=-1,
    what='total comments')
plot_activity_pie.__doc__ = """
    Plots pie-chart of comment_activity of commenters is project.
    'what' can be 'total comments', or 'word counts'"""

plot_comment_histogram = partial(
    plot_from_network, "plot_author_activity_hist",
    thread_type="all threads", stage=-1,
    what="total comments", bins=10, fontsize=6)
plot_comment_histogram.__doc__ = "Plots histogram of commenting activity."

plot_scatter_authors = partial(
    plot_from_network_tp, "scatter_authors",
    thread_type="all threads", stage=-1,
    thread=None,
    measure="betweenness centrality",
    weight=(None, None),
    thresh=15,
    xlim=None, ylim=None)
plot_scatter_authors.__doc__ = """
        Scatter-plot with position based on interaction and cluster
        measure, color based on number of comments, and size on avg comment
        length.
        If thread is not None, an author_network is created for single thread
        at iloc[thread]"""

plot_scatter_authors_hits = partial(
    plot_from_network_tp, "scatter_authors_hits",
    thread_type="all threads", stage=-1,
    thread=None, thresh=15)
plot_scatter_authors_hits.__doc__ = """
    Scatter-plot based on hits-algorithm for hubs and authorities."""

plot_scatter_comments_replies = partial(
    plot_from_network, "scatter_comments_replies",
    thread_type="all threads", stage=-1)
plot_scatter_comments_replies.__doc__ = """
    Scatter-plot of comments vs direct replies received"""

plot_interaction_trajectories = partial(
    plot_from_network_tp, "plot_i_trajectories",
    thread_type="all threads", stage=-1,
    thread=None, loops=False,
    thresh=None, select=None, l_thresh=5)
plot_interaction_trajectories.__doc__ = """
    Plots interaction-trajectories for each pair of contributors."""

plot_delays_boxplot = partial(
    plot_from_network_tp, "plot_centre_closeness",
    thread_type="all threads", stage=-1,
    thread=None, thresh=10, ylim=16)
plot_delays_boxplot.__doc__ = """
    Boxplot of time before return to centre for core authors"""

plot_distance_from_centre = partial(
    plot_from_network_tp, "plot_centre_dist",
    thread_type="all threads", stage=-1,
    thread=None, thresh=2,
    show_threads=True)
plot_distance_from_centre.__doc__ = """
    Plots time elapsed since last comment for each participant"""

plot_centre_crowd = partial(
    plot_from_network_tp, "plot_centre_crowd",
    thread_type="all threads", stage=-1,
    thread=None, thresh=2,
    show_threads=False)
plot_centre_crowd.__doc__ = """
    Plotting evolution of number of participants close to centre"""


def draw_network(pm_frame, project, **kwargs):
    """Draws and shows author_network of chosen type"""
    thread_type = kwargs.pop("thread_type", "all threads")
    stage = kwargs.pop("stage", -1)
    kwargs['project'] = project
    draw_author_network(
        get_project_at(pm_frame, project, thread_type, stage)['network'],
        **kwargs)


def plot_edge_removal(pm_frame, project, **kwargs):
    """Plots effect on components and network-measures of
    gradually removing weak links."""
    thread_type = kwargs.pop("thread_type", "all threads")
    stage = kwargs.pop("stage", -1)
    kwargs['project'] = project
    shrinking_components_edges(
        get_project_at(pm_frame, project, thread_type, stage)['network'],
        **kwargs)


def plot_node_removal(pm_frame, project, **kwargs):
    """Plots effect on components and network-measures of
    gradually removing strongly connected nodes."""
    thread_type = kwargs.pop("thread_type", "all threads")
    stage = kwargs.pop("stage", -1)
    kwargs['project'] = project
    shrinking_components_nodes(
        get_project_at(pm_frame, project, thread_type, stage)['network'],
        **kwargs)


draw_centre = partial(
    plot_from_network, "draw_centre_discussion",
    thread_type="all threads", stage=-1,
    show=False, skips=10, zoom=1)
draw_centre.__doc__ = "Animated plot of who is close to centre"
