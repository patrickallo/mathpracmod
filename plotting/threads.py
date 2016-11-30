"""module with wrapper-functions to access plotting-functions from
MultiCommentThread"""

from notebook_helper.access_funs import get_project_at


def plot_activity_author(pm_frame, project,
                         thread_type="all threads", stage=-1,
                         intervals=1):
    """Wrapper function for mthread.plot_activity_author
    Plots author activity over time for project (limited to thread_type)"""
    data = get_project_at(
        pm_frame, project, thread_type, stage)['mthread (accumulated)']
    data.plot_activity_author(project=project,
                              intervals=intervals)


def plot_activity_thread(pm_frame, project,
                         thread_type="all threads", stage=-1,
                         color_by="cluster",
                         intervals=1):
    """Wrapper function for mthread.plot_activity_thread
    Plots thread activity over time for project (limited to thraead_type).
    color_by is either cluster or author"""
    data = get_project_at(
        pm_frame, project, thread_type, stage)['mthread (accumulated)']
    data.plot_activity_thread(project=project,
                              color_by=color_by,
                              intervals=intervals)


def plot_discussion_tree(pm_frame, project,
                         thread_type="all threads", stage=-1,
                         intervals=10):
    """Wrapper function for mthread.draw_graph
    Plots structure of discussion in project"""
    data = get_project_at(
        pm_frame, project, thread_type, stage)['mthread (accumulated)']
    data.draw_graph(project=project,
                    intervals=intervals)


# could be extended to single mthreads to look at growth of single threads
def plot_growth(pm_frame, project,
                thread_type="all threads", stage=-1):
    """Wrapper function for mthread.plot_growth
    Plots growth in comments in discussion"""
    data = get_project_at(
        pm_frame, project, thread_type, stage)['mthread (accumulated)']
    data.plot_growth(project=project)
