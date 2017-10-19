"""module with wrapper-functions to access plotting-functions from
MultiCommentThread"""

from functools import partial
from operator import methodcaller
from numpy import full
from pandas import DataFrame, Series
from seaborn import swarmplot
from notebook_helper.access_funs import get_project_at
from plotting.graphs import draw_discussion_tree, draw_discussion_tree_radial
from access_classes import show_or_save


def plot_from_mthread(plot_method, pm_frame, project, **kwargs):
    """Auxiliary-function calling supplied plot_method of data"""
    data = get_project_at(pm_frame, project, kwargs["thread_type"],
                          kwargs["stage"])["mthread (accumulated)"]
    methodcaller(plot_method, project=project, **kwargs)(data)


# plot_discussion_tree = partial(plot_from_mthread, "draw_graph",
#                                thread_type="all threads", stage=-1,
#                                intervals=10)
# plot_discussion_tree.__doc__ = "Plots discussion-structure from project"


def plot_discussion_tree(pm_frame, project, **kwargs):
    """Plots tree-like discussion-tree"""
    thread_type = kwargs.pop("thread_type", "all threads")
    stage = kwargs.pop("stage", -1)
    kwargs["project"] = project
    draw_discussion_tree(
        get_project_at(
            pm_frame, project, thread_type, stage)["mthread (accumulated)"],
        **kwargs)


def plot_discussion_tree_radial(pm_frame, project, **kwargs):
    """Plots radial discussion-tree"""
    thread_type = kwargs.pop("thread_type", "all threads")
    stage = kwargs.pop("stage", -1)
    kwargs["project"] = project
    draw_discussion_tree_radial(
        get_project_at(
            pm_frame, project, thread_type, stage)["mthread (accumulated)"],
        **kwargs)


def plot_threads_swarm(pm_frame, project, **kwargs):
    """Plots swarm-plot of timestamps and colors per author or episode"""
    thread_type = kwargs.pop("thread_type", "all threads")
    show = kwargs.pop("show", True)
    stage = kwargs.pop("stage", -1)
    if isinstance(stage, int):
        stage = slice(stage)
    color_by = kwargs.pop("color_by", "cluster")
    data = pm_frame.loc[project][thread_type, "mthread (single)"].iloc[
        stage]
    plot_data = DataFrame()
    for i, data in data.iteritems():
        time_index, time_data, time_authors, time_cluster = zip(
            *[(com_id,
               mdata['com_timestamp'],
               mdata['com_author'],
               mdata['cluster_id'][0]) for com_id, mdata in
              data.graph.nodes_iter(data=True)])
        time_data = Series(time_data, index=time_index)
        time_data = time_data.sort_values()
        time_data = Series(time_data - time_data[0]).astype(int)
        time_data = DataFrame(
            {'time': time_data,
             'authors': Series(time_authors, index=time_index),
             'cluster': Series(time_cluster, index=time_index)})
        time_data["colors"] = [data.author_color[author] for
                               author in time_data['authors']]
        time_data["authors"] = time_data["authors"].astype("category")
        time_data["colors"] = time_data["colors"].astype("category")
        time_data["cluster"] = time_data["cluster"].astype("category")
        time_data["threads"] = full(len(time_data.index), i)
        plot_data = plot_data.append(time_data)
    title = "Threads in {} by {}".format(project, color_by)
    color_by = "colors" if color_by == "author" else color_by
    axes = swarmplot(
        x="threads", y="time", hue=color_by,
        data=plot_data, palette="tab20")
    axes.legend_.remove()
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.set_title(title)
    show_or_save(show)


plot_activity_thread = partial(plot_from_mthread, "plot_activity_thread",
                               thread_type="all threads", stage=-1,
                               color_by="cluster",
                               intervals=1)
plot_activity_thread.__doc__ = """Plots thread activity over time for project
                                  (limited to thraead_type).
                                  color_by is either cluster or author"""

plot_activity_author = partial(plot_from_mthread, "plot_activity_author",
                               thread_type="all threads", stage=-1,
                               intervals=1)
plot_activity_author.__doc__ = """Plots author activity over time for project
                                  (limited to thread_type)"""

plot_growth_size = partial(plot_from_mthread, "plot_growth_size",
                           thread_type="all threads", stage=-1)
plot_growth_size.__doc__ = "Plots barplot of comments per week."

# could be extended to single mthreads to look at growth of single threads
plot_growth = partial(plot_from_mthread, "plot_growth",
                      thread_type="all threads", stage=-1)
plot_growth.__doc__ = "Plots growth in comments in discussion"

plot_comment_sizes = partial(plot_from_mthread, "plot_sizes",
                             thread_type="all threads", stage=-1,
                             resample="Daily")
plot_comment_sizes.__doc__ = """Plots average wordcounts of comments per
                                resample (default=Daily)."""
