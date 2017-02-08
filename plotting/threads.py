"""module with wrapper-functions to access plotting-functions from
MultiCommentThread"""

from functools import partial
from operator import methodcaller
from notebook_helper.access_funs import get_project_at


def plot_from_mthread(plot_method, pm_frame, project, **kwargs):
    """Auxiliary-function calling supplied plot_method of data"""
    data = get_project_at(pm_frame, project, kwargs["thread_type"],
                          kwargs["stage"])["mthread (accumulated)"]
    methodcaller(plot_method, project=project, **kwargs)(data)


plot_discussion_tree = partial(plot_from_mthread, "draw_graph",
                               thread_type="all threads", stage=-1,
                               intervals=10)
plot_discussion_tree.__doc__ = "Plots discussion-structure from project"

plot_activity_author = partial(plot_from_mthread, "plot_activity_author",
                               thread_type="all threads", stage=-1,
                               intervals=1)
plot_activity_author.__doc__ = """Plots author activity over time for project
                                  (limited to thread_type)"""

plot_activity_thread = partial(plot_from_mthread, "plot_activity_thread",
                               thread_type="all threads", stage=-1,
                               color_by="cluster",
                               intervals=1)
plot_activity_thread.__doc__ = """Plots thread activity over time for project
                                  (limited to thraead_type).
                                  color_by is either cluster or author"""

# could be extended to single mthreads to look at growth of single threads
plot_growth = partial(plot_from_mthread, "plot_growth",
                      thread_type="all threads", stage=-1)
plot_growth.__doc__ = "Plots growth in comments in discussion"

plot_growth_size = partial(plot_from_mthread, "plot_growth_size",
                           thread_type="all threads", stage=-1)
plot_growth_size.__doc__ = "Plots barplot of comments per week."
