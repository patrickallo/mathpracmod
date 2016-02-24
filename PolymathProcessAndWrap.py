"""
Module with script to process Polymath-projects, and wrapper-functions for
methods of comment_thread and author_network objects.
Intended use: in notebooks.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

from author_network import AuthorNetwork
from comment_thread import *

def process_polymath(project, split=False):
    """Takes name of a PM-project and returns pandas DataFrame"""
    message = "Polymath {}".format(
        project.split(" ")[-1]) if project.startswith("pm") else "Mini\
        Polymath {}".format(project[-1])
    settings = {
        'msg': message,
        'filename': message.replace(" ", ""),
        'source': project.replace(" ", ""),
        'urls': [],
        'type': '',
        'parser': 'html5lib',
        'cmap': plt.cm.Paired, # not a string, this is an object
        'vmin': 1,
        'vmax': 100}

    with open("DATA/" + settings['source'] + ".csv", "r") as input_file:
        pm_frame = pd.read_csv(input_file, index_col="Ord")

    pm_frame['blog'] = pm_frame['url'].apply(
        lambda url: urlparse(url).netloc.split('.')[0].title())
    pm_frame['thread'] = [THREAD_TYPES[blog](url) for (url, blog) in zip(
        pm_frame['url'], pm_frame['blog'])]
    pm_frame['number of comments'] = pm_frame['thread'].apply(
        lambda x: len(x.node_name.keys()))
    pm_frame['number of comments (accumulated)'] = pm_frame[
        'number of comments'].cumsum()

    indices = pm_frame.index
    threads = pm_frame.thread
    pm_frame['mthread (single)'] = pm_frame['thread'].apply(MultiCommentThread)
    pm_frame['mthread (accumulated)'] = Series(
        [MultiCommentThread(*threads[0:i+1]) for i in indices], index=indices)
    pm_frame['network'] = pm_frame['mthread (accumulated)'].apply(
        AuthorNetwork)

    if split:
        r_indices = pm_frame[pm_frame['research']].index
        d_indices = pm_frame[~pm_frame['research']].index
        r_threads = pm_frame[pm_frame['research']].thread
        d_threads = pm_frame[~pm_frame['research']].thread
        pm_frame['r_mthread (accumulated)'] = Series(
            [MultiCommentThread(*r_threads[0:i+1]) for i in r_indices],
            index=r_indices)
        pm_frame['d_mthread (accumulated)'] = Series(
            [MultiCommentThread(*d_threads[0:i+1]) for i in d_indices],
            index=d_indices)
        pm_frame['r_network'] = pm_frame[
            pm_frame['research']]['r_mthread (accumulated)'].apply(
                AuthorNetwork)
        pm_frame['d_network'] = pm_frame[
            ~pm_frame['research']]['d_mthread (accumulated)'].apply(
                AuthorNetwork)
        pm_frame = pm_frame.reindex_axis(['title', 'url', 'blog', 'research',
                                          'number of comments',
                                          'number of comments (accumulated)',
                                          'thread', 'mthread (single)',
                                          'mthread (accumulated)',
                                          'network',
                                          'r_mthread (accumulated)',
                                          'r_network',
                                          'd_mthread (accumulated)',
                                          'd_network'],
                                         axis=1)
    else:
        pm_frame = pm_frame.reindex_axis(['title', 'url', 'blog', 'research',
                                          'number of comments',
                                          'thread', 'mthread (single)',
                                          'mthread (accumulated)', 'network'],
                                         axis=1)

    pm_frame.index = pd.MultiIndex.from_tuples([(settings['msg'], i) for i in indices],
                                               names=['Project', 'Ord'])

    return pm_frame


def get_project_at(project, stage):
    """Helper function that gets a project from a frame.
    Only works if the frame is defined outside the function."""
    return mPM_FRAME.loc[project].iloc[stage] if project.startswith(
        "Mini") else PM_FRAME.loc[project].iloc[stage]

def draw_network(project, stage=-1):
    """Wrapper function for author_network.draw_graph.
    Plots the interaction-network between the commenters in project."""
    get_project_at(project, stage)['network'].draw_graph(project=project)

def draw_centre(project, stage=-1, skips=10, zoom=1):
    """Wrapper function for author_network.draw_centre_discussion"""
    get_project_at(project, stage)['network'].draw_centre_discussion(
        skips=skips, zoom=zoom)

def plot_activity_pie(project, stage=-1):
    """Wrapper function for author_network.plot_author_activity_pie
    Plots pie-chart of comment_activity of commenters is project."""
    get_project_at(project, stage)['network'].plot_author_activity_pie(
        project=project)

def plot_activity_bar(project, stage=-1):
    """Wrapper function for author_network.plot_activity_bar
    Plots bar-chart of comment_activity of commenters in project"""
    get_project_at(project, stage)['network'].plot_author_activity_bar(
        project=project)

def plot_degree_centrality(project, stage=-1):
    """Wrapper function for author_network.plot_degree_centrality
    Plots line-chart of degree-centrality of commenters in project"""
    get_project_at(project, stage)['network'].plot_degree_centrality(
        project=project)

def plot_activity_degree(project, stage=-1):
    """Wrapper function for author_network.plit_activity_degree
    Plots superposition of bar-chart of comment-activity and line-chart of degree-centrality"""
    get_project_at(project, stage)['network'].plot_activity_degree(
        project=project)

def plot_discussion(project, intervals=10, first=SETTINGS['first_date'],
                    last=SETTINGS['last_date'], stage=-1):
    """Wrapper function for mthread.draw_graph
    Plots structure of discussion in project"""
    get_project_at(project, stage)['mthread (accumulated)'].draw_graph(
        intervals=intervals,
        first=first,
        last=last,
        project=project)

def plot_activity(project, intervals=1, first=SETTINGS['first_date'], last=SETTINGS['last_date'],
                  activity='thread', stage=-1):
    """Wrapper function for mthread.plot_activity
    Plots thread or author activity over time for project"""
    get_project_at(project, stage)['mthread (accumulated)'].plot_activity(
        activity, intervals=intervals,
        first=first,
        last=last,
        project=project)

def plot_growth(project, last=datetime.today(), stage=-1):
    """Wrapper function for mthread.plot_growth
    Plots growth in comments in discussion"""
    get_project_at(project, stage)['mthread (accumulated)'].plot_growth(
        project=project, last=last)

def get_last(lst_of_frames):
    """Helper function which returns the final line for each project
    from a DataFrame"""
    if lst_of_frames == POLYMATHS:
        source = PM_FRAME
        positions = np.array(
            [frame.index.levels[1][-1] for frame in lst_of_frames]
            ).cumsum() + np.arange(len(POLYMATHS))
    elif lst_of_frames == MINIPOLYMATHS:
        source = mPM_FRAME
        positions = np.array(
            [frame.index.levels[1][-1] for frame in lst_of_frames]
            ).cumsum() + np.arange(len(MINIPOLYMATHS))
    else:
        raise ValueError("Need either POLYMATHS or MINIPOLYMATHS")
    source.index = source.index.swaplevel(0, 1)
    data = source.iloc[positions]
    source.index = source.index.swaplevel(1, 0)
    data.index = data.index.droplevel()
    return data, positions

def plot_thread_engagement(project, compress=1, sel=[]):
    """Takes a project as argument, and shows four types of data for each
    thread in bar-plot:
    - y-axis: the average number of comments per participant
        (a measure of how diverse each thread is)
    - width of each bar: the number of comments
    - bar-color: the type of thread
    - text above each bar included in the optional kwarg sel: number of
        comments and number of participants."""
    data = mPM_FRAME.loc[project] if project.startswith("Mini") else PM_FRAME.loc[project]
    authors = data['authors'].apply(len)
    engagement = authors / data['number of comments']
    df = DataFrame({'research threads': engagement[data['research']],
                    'discussion threads': engagement[~data['research']]},
                   #columns = ['research threads', 'discussion threads'],
                   index=data.index)
    sizes = (data['number of comments'] / compress).tolist()
    df.index.name = "Threads"
    matplotlib.style.use('seaborn-notebook')
    fig = plt.figure()
    axes = df.plot(kind='bar', color=['lightsteelblue', 'steelblue'],
                   title="Community engagement in {}".format(project))
    axes.set_ylabel('average number of comments per participant')
    axes.set_yticklabels([round(1/i, 2) for i in axes.get_yticks()])
    axes.set_xticklabels(data['title'].apply(lambda x: x[:40]), rotation=90, fontsize='small')
    for container in axes.containers:
        for i, child in enumerate(container.get_children()):
            child.set_x(df.index[i] - sizes[i]/2)
            plt.setp(child, width=sizes[i])
    for i in engagement.index:
        if i in sel:
            axes.text(engagement.index[i] + .2, engagement[i],
                      "{} comments\n {} commenters".format(
                          data['number of comments'][i], authors[i]),
                      ha="center", va="bottom", fontsize='small')
    plt.tight_layout()

def plot_thread_evolution(project, compress=1, sel=[], sharex=True):
    """takes a project as argument and shows two plots:
    - The bar-plot thread_engagement described above
    - The evolution of the number of participants in each thread (active,
    joined, left) as an area-plot"""
    # data for evolution
    data = mPM_FRAME.loc[project] if project.startswith(
        "Mini") else PM_FRAME.loc[project]
    added = (data['authors'] - data['authors'].shift(1)).apply(
        lambda x: 0 if isinstance(x, float) else len(x))
    removed = (data['authors'].shift(1) - data['authors']).apply(
        lambda x: 0 if isinstance(x, float) else - len(x))
    size = data['authors'].apply(len) - added
    df1 = DataFrame({'joined' : added, 'left' : removed, 'current': size},
                    columns=["joined", "current", "left"], index=data.index)
    df1.index.name = "Threads"
    # data for engagement
    authors = data['authors'].apply(len)
    engagement = authors / data['number of comments']
    df2 = DataFrame({'research threads': engagement[data['research']],
                     'discussion threads': engagement[~data['research']]},
                    #columns = ['research threads', 'discussion threads'],
                    index=data.index)
    sizes = (data['number of comments'] / compress).tolist()
    df2.index.name = "Threads"
    # setting up plot
    matplotlib.style.use('seaborn-talk')
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), squeeze=False, sharex=sharex)
    plt.subplots_adjust(hspace=0.2)
    # plot bottom
    df1.plot(kind="area", ax=axes[1][0], title="",
             color=['sage', 'lightgrey', 'indianred'], stacked=True)
    axes[1][0].set_xticks(df1.index)
    axes[1][0].set_xticklabels(data['title'], rotation=90, fontsize='small')
    axes[1][0].set_xlabel("")
    axes[1][0].set_ylabel('active commenters')
    # plot top
    df2.plot(kind='bar', ax=axes[0][0], color=['lightsteelblue', 'steelblue'],
             title="Community engagement in {}".format(project))
    axes[0][0].set_ylabel('comments per participant')
    axes[0][0].set_yticklabels([round(1/i, 2) for i in axes[0][0].get_yticks()])
    axes[0][0].yaxis.get_major_ticks()[0].label1.set_visible(False)
    axes[0][0].set_xticklabels(df2.index, fontsize='small')
    axes[0][0].set_xlabel("")
    for container in axes[0][0].containers:
        for i, child in enumerate(container.get_children()):
            child.set_x(df2.index[i] - sizes[i]/2)
            plt.setp(child, width=sizes[i])
    for i in engagement.index:
        if i in sel:
            axes[0][0].text(engagement.index[i] + .2, engagement[i],
                            "{} comments\n {} commenters".format(
                                data['number of comments'][i], authors[i]),
                            ha="center", va="bottom", fontsize='small')

def plot_community_evolution(project):
    """Takes a project or group of projects ("Polymath" of "MiniPolymath") as
    an argument, and shows the evolution of the number of participants per
    thread or per project in the same manner as the area-plot in
    plot_thread_evolution."""
    if isinstance(project.split()[-1], int):
        as_threads = True
        data = mPM_FRAME.loc[project] if project.startswith(
            "Mini") else PM_FRAME.loc[project]
        added = (data['authors'] - data['authors'].shift(1)).apply(
            lambda x: 0 if isinstance(x, float) else len(x))
        removed = (data['authors'].shift(1) - data['authors']).apply(
            lambda x: 0 if isinstance(x, float) else - len(x))
        size = data['authors'].apply(len) - added
        df = DataFrame({'joined' : added, 'left' : removed, 'current': size},
                       columns=["joined", "current", "left"], index=data.index)
        df.index.name = "Threads"
    else:
        as_threads = False
        if project.startswith("Mini"):
            data, positions = get_last(MINIPOLYMATHS)
        elif project.startswith("Poly"):
            data, positions = get_last(POLYMATHS)
        else:
            raise ValueError("Need either Polymath or Mini Polymath")
        added = (data['authors (accumulated)'] -
                 data['authors (accumulated)'].shift(1)).apply(
                     lambda x: 0 if isinstance(x, float) else len(x))
        removed = (data['authors (accumulated)'].shift(1) -
                   data['authors (accumulated)']).apply(
                       lambda x: 0 if isinstance(x, float) else - len(x))
        size = data['authors (accumulated)'].apply(len) - added
        df = DataFrame({'joined' : added, 'left' : removed, 'current': size},
                       columns=["joined", "current", "left"])
        df.index = list(range(1, len(positions) + 1))

    matplotlib.style.use('seaborn-notebook')
    fig = plt.figure()
    axes = df.plot(kind="area", title="Community Evolution in {}".format(
        project), color=['sage', 'lightgrey', 'indianred'], stacked=True)
    axes.set_xticks(df.index)
    if as_threads:
        axes.set_xticklabels(data['title'], rotation=90, fontsize='small')
    else:
        xlabels = sorted(data.index, key=lambda x: int(x.split()[-1]))
        axes.set_xticklabels(xlabels, rotation=90, fontsize='small')
    axes.set_ylabel('number of active commenters')

def plot_participation_evolution(project, n=2, skip_anon=True):
    """Takes a project or group of projects ("Polymath" of "MiniPolymath") as
    an argument, and shows for each participant that took part in at least n
    projects/threads the projects/threads (s)he participated in."""
    if project.split()[-1]in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
        print("process threads")
        as_threads = True
        if project.startswith("Mini"):
            data = mPM_FRAME.loc[project]
        elif project.startswith("Poly"):
            data = PM_FRAME.loc[project]
        else:
            raise ValueError("Need either Polymath or Mini Polymath project")
        all_authors = data.iloc[-1]['authors (accumulated)']
        data = data['authors']
        title = "Participation per thread in " + project
    else:
        as_threads = False
        if project.startswith("Mini"):
            data, positions = get_last(MINIPOLYMATHS)
            all_authors = list(ALL_MINI_AUTHORS)
            title = "Participation per project in Mini Polymath"
        elif project.startswith("Poly"):
            data, positions = get_last(POLYMATHS)
            all_authors = list(ALL_AUTHORS)
            title = "Participation per project in Polymath"
        else:
            raise ValueError("Need either Polymath or Mini Polymath")
        data = data['authors (accumulated)']
    indices = data.index.tolist()
    author_project = DataFrame(index=all_authors)
    for ind in indices:
        author_project[ind] = np.zeros_like(author_project.index, dtype=bool)
        for author in data[ind]:
            author_project[ind][author] = True
    author_project = author_project.sort_values(by=indices, ascending=False)
    author_project = author_project.drop("Anonymous") if skip_anon else author_project
    select = author_project.sum(axis=1) >= n
    matplotlib.style.use('seaborn-notebook')
    factor = 30 - len(indices) if len(indices) <= 30 else 40 - len(indices)
    colors = [plt.cm.Set1(factor*i) for i in range(len(indices))]
    author_project.loc[select].plot(kind="bar", stacked=True, color=colors,
                                    title=title)
