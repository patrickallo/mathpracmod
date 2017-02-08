"""module with wrapper-functions to access plotting-functions from
AuthorNetwork"""
# imports
from notebook_helper.access_funs import get_project_at, thread_or_project


def plot_activity_pie(pm_frame, project,
                      thread_type="all threads", stage=-1,
                      what='total comments'):
    """Wrapper function for author_network.plot_author_activity_pie
    Plots pie-chart of comment_activity of commenters is project.
    'what' can be 'total comments', or 'word counts'"""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    data.plot_author_activity_pie(project=project,
                                  what=what)


def plot_activity_bar(pm_frame, project,
                      thread_type="all threads", stage=-1,
                      what="by level",
                      fontsize=6):
    """Wrapper function for author_network.plot_activity_bar
    Plots bar-chart of comment_activity of commenters in project
    'what' can be 'by level' or 'word counts'"""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    data.plot_author_activity_bar(project=project,
                                  what=what,
                                  fontsize=fontsize)


def plot_activity_prop(pm_frame, project,
                       thread_type="all threads", stage=-1):
    """Wrapper function for author_network.plot_activity_prop"""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    data.plot_activity_prop(project=project)


def plot_activity_degree(pm_frame, project,
                         thread_type="all threads", stage=-1,
                         graph_type='interaction',
                         measures=None, weight=None,
                         delete_on=None, thresh=0,
                         fontsize=6):
    """Wrapper function for author_network.plot_activity_degree
    Plots superposition of bar-chart of comment-activity and line-chart of
    degree-centrality"""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    data.plot_activity_degree(project=project,
                              g_type=graph_type,
                              measures=measures, weight=weight,
                              delete_on=delete_on, thresh=thresh,
                              fontsize=fontsize)


def plot_centrality_measures(pm_frame, project,
                             thread_type="all threads", stage=-1,
                             graph_type='interaction',
                             measures=None,
                             delete_on=None, thresh=0,
                             fontsize=6):
    """Wrapper function for author_network.centrality_measures
    Plots line-chart of different centrality-measures."""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    data.plot_centrality_measures(project=project,
                                  g_type=graph_type,
                                  measures=measures,
                                  delete_on=delete_on, thresh=thresh,
                                  fontsize=fontsize)


def plot_comment_histogram(pm_frame, project,
                           thread_type="all threads", stage=-1,
                           what="total comments", bins=10,
                           fontsize=6):
    """Wrapper function for author_network.plot_author_activity_hist.
    'what' is either 'total comments' or 'word counts'"""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    data.plot_author_activity_hist(project=project,
                                   what=what, bins=bins,
                                   fontsize=fontsize)


def plot_interaction_trajectories(pm_frame, project,
                                  thread_type="all threads", stage=-1,
                                  thread=None,
                                  loops=False,
                                  thresh=None, select=None,
                                  l_thresh=5):
    """Wrapper function for author_network.plot_i_trajectories.
    If thread is not None, an author_network is created for a single thread at
    iloc[thread]"""
    project, data = thread_or_project(
        pm_frame, project, thread_type, stage, thread)
    data.plot_i_trajectories(project=project,
                             loops=loops, thresh=thresh, select=select,
                             l_thresh=l_thresh)


def plot_delays(pm_frame, project,
                thread_type="all threads", stage=-1,
                thread=None,
                thresh=10,
                ylim=16):
    """Wrapper function for author_network.plot_centre_closeness.
    If thread is not None, an author_network is created for a single thread at
    iloc[thread]"""
    project, data = thread_or_project(
        pm_frame, project, thread_type, stage, thread)
    data.plot_centre_closeness(project=project,
                               thresh=thresh, ylim=ylim)


def plot_distance_from_centre(pm_frame, project,
                              thread_type="all threads", stage=-1,
                              thread=None,
                              thresh=2,
                              show_threads=True):
    """Wrapper function for author_network.plot_centre_dist.
    If thread is not None, an author_network is created for single thread at
    iloc[thread]"""
    project, data = thread_or_project(
        pm_frame, project, thread_type, stage, thread)
    data.plot_centre_dist(project=project,
                          thresh=thresh, show_threads=show_threads)


def plot_centre_crowd(pm_frame, project,
                      thread_type="all threads", stage=-1,
                      thread=None,
                      thresh=2,
                      show_threads=False):
    """Wrapper function for author_network.plot_centre_crowd.
    If thread is not None, an author_network is created for single thread at
    iloc[thread]"""
    project, data = thread_or_project(
        pm_frame, project, thread_type, stage, thread)
    data.plot_centre_crowd(project=project,
                           thresh=thresh, show_threads=show_threads)


def plot_scatter_authors(pm_frame, project,
                         thread_type="all threads", stage=-1,
                         thread=None,
                         measure="betweenness centrality",
                         weight=(None, None),
                         thresh=15):
    """Wrapper function for author_network.scatter_authors.
    If thread is not None, an author_network is created for single thread at
    iloc[thread]"""
    project, data = thread_or_project(
        pm_frame, project, thread_type, stage, thread)
    data.scatter_authors(measure=measure, weight=weight,
                         project=project, thresh=thresh)


def plot_scatter_authors_hits(pm_frame, project,
                              thread_type="all threads", stage=-1,
                              thread=None,
                              thresh=15):
    """Wrapper function for author_network.scatter_authors_hits.
    If thread is not None, an author_network is created for single thread at
    iloc[thread]"""
    project, data = thread_or_project(
        pm_frame, project, thread_type, stage, thread)
    data.scatter_authors_hits(project=project, thresh=thresh)


def measures_corr(pm_frame, project,
                  graph_type, weight=None,
                  thread_type="all threads", stage=-1):
    """wrapper function for correlation between network-measures in
    author_network"""
    data = get_project_at(pm_frame, project, thread_type, stage)['network']
    return data.corr_centrality_measures(g_type=graph_type, weight=weight)


def draw_network(pm_frame, project,
                 graph_type,
                 thread_type="all threads", stage=-1,
                 reset=False):
    """Wrapper function for author_network.draw_graph.
    Plots the interaction-network between the commenters in project."""
    get_project_at(
        pm_frame, project, thread_type, stage)['network'].draw_graph(
            project=project, graph_type=graph_type, reset=reset)


def draw_centre(pm_frame, project,
                thread_type="all threads", stage=-1,
                show=False, skips=10, zoom=1):
    """Wrapper function for author_network.draw_centre_discussion"""
    get_project_at(pm_frame, project,
                   thread_type, stage)['network'].draw_centre_discussion(
                       show=show, skips=skips, zoom=zoom)
