"""Module with plotting-functions to generate overview-plots
based on a pm_frame with all data from the PM-projects"""
# imports
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns

from access_classes import fake_legend, get_len_v
from author_network import SETTINGS
from plotting.heatmap import general_heatmap
from notebook_helper.access_funs import get_last

SBSTYLE = SETTINGS['style']

sns.set_style('white')


def _added_removed(data, thread_type, authors):
    """helper-fun called by plot_community_evolution and
    plot_thread_evolution to compute differences between
    sets of participants in a data-frame"""
    helper_p = np.vectorize(lambda x: 0 if isinstance(x, float) else len(x))
    helper_n = np.vectorize(lambda x: 0 if isinstance(x, float) else -len(x))
    added = helper_p(data[thread_type, authors] -
                     data[thread_type, authors].shift(1))
    removed = helper_n(data[thread_type, authors].shift(1) -
                       data[thread_type, authors])
    return added, removed


def plot_overview(pm_frame, annotate=True):
    """Combined bar-plot and line-plot of number of comments and
    number of participants in all projects"""
    the_data, _ = get_last(pm_frame, None)
    the_data.index = the_data.index.droplevel(1)
    the_data.columns = the_data.columns.swaplevel()
    author_data = the_data['authors (accumulated)'].copy()
    comment_data = the_data[
        'number of comments (accumulated)', 'all threads'].copy()
    author_data['authors only active in research threads'] = author_data[
        'research threads'] - author_data['discussion threads']
    author_data['authors only active in "discussion" threads'] = author_data[
        'discussion threads'] - author_data['research threads']
    author_data[
        'authors active in both types of threads'] = author_data[
            'all threads'] - author_data[
                'authors only active in research threads'] - author_data[
                    'authors only active in "discussion" threads']
    for project in author_data.index:
        if pd.isnull(author_data.loc[project][
                'authors only active in research threads']):
            author_data.loc[project][
                'authors only active in research threads'] = author_data.loc[
                    project]['all threads']
    author_data = author_data[
        ['authors only active in research threads',
         'authors active in both types of threads',
         'authors only active in "discussion" threads']]
    author_data = author_data.applymap(
        lambda set: len(set) if pd.notnull(set) else 0)
    mpl.style.use(SBSTYLE)
    axes = plt.subplot()
    author_data.plot(
        kind='bar', stacked=True,
        color=['steelblue', 'lightsteelblue', 'lightgrey'], ax=axes)
    # title="Overview of all projects")
    axes.xaxis.set_ticks_position('bottom')
    axes.set_ylabel("Number of participants")
    axes.set_ylim(0, 200)
    axes.set_yticks(range(0, 200, 25))
    axes.legend(loc=2)
    if annotate:
        y_values = author_data.sum(axis=1).loc[
            ["Polymath {}".format(i) for i in [1, 4, 5, 8, 11, 13, 14]]].values
        axes.annotate(
            'published', xy=(0, y_values[0]), xytext=(0, y_values[0] + 20),
            arrowprops=dict(facecolor='steelblue', shrink=0.05),
            horizontalalignment='center')
        axes.annotate(
            'published', xy=(3, y_values[1]), xytext=(3, y_values[1] + 20),
            arrowprops=dict(facecolor='steelblue', shrink=0.05),
            horizontalalignment='center')
        axes.annotate(
            're-used', xy=(4, y_values[2]), xytext=(4, y_values[2] + 20),
            arrowprops=dict(facecolor='lightsteelblue', shrink=0.05),
            horizontalalignment='center')
        axes.annotate(
            'published', xy=(7, y_values[3]), xytext=(7.5, y_values[3] + 10),
            arrowprops=dict(facecolor='steelblue', shrink=0.05),)
        axes.annotate(
            'closed with partial results', xy=(10, y_values[4]),
            xytext=(10, y_values[4] + 20),
            arrowprops=dict(facecolor='steelblue', shrink=0.05),
            horizontalalignment='center')
        axes.annotate(
            'first success booked', xy=(12, y_values[5]),
            xytext=(12, y_values[5] + 20),
            arrowprops=dict(facecolor='steelblue', shrink=0.05),
            horizontalalignment='center')
        axes.annotate(
            'results submitted', xy=(13, y_values[6]),
            xytext=(13, y_values[6]+22),
            arrowprops=dict(facecolor='steelblue', shrink=0.05),
            horizontalalignment='left')
    comment_data = np.sqrt(comment_data)
    axes2 = axes.twinx()
    axes2.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "{:0.0f}".format(np.square(x))))
    axes2.set_ylabel("Number of comments")
    axes2.plot(axes.get_xticks(), comment_data.values,
               linestyle='-', marker='.', linewidth=.5,
               color='darkgrey')
    plt.savefig("FIGS/overview_bar.png")


def plot_comments_boxplot(pm_frame):
    """Create box-plot of commenting-activity"""
    commenting_author_project_r = get_last(
        pm_frame,
        "research threads")[0][
            'research threads', 'comment_counter (accumulated)']
    commenting_author_project_r.index =\
        commenting_author_project_r.index.droplevel(1)
    commenting_author_project_d = get_last(
        pm_frame,
        "discussion threads")[0][
            'discussion threads', 'comment_counter (accumulated)']
    commenting_author_project_d.index =\
        commenting_author_project_d.index.droplevel(1)
    commenting_author_project_a = get_last(
        pm_frame,
        "all threads")[0][
            'all threads', 'comment_counter (accumulated)']
    commenting_author_project_a.index =\
        commenting_author_project_a.index.droplevel(1)
    commenting_author_project_r = commenting_author_project_r.apply(Series).T
    commenting_author_project_d = commenting_author_project_d.apply(Series).T
    commenting_author_project_a = commenting_author_project_a.apply(Series).T
    _, axes = plt.subplots(1, 3, figsize=(12, 6))
    mpl.rc("lines", markeredgewidth=0.3)
    commenting_author_project_a.apply(Series).plot(
        kind='box', ax=axes[0], grid=False, logy=True, sym='.',
        rot=90, return_type='axes', color='steelblue',
        title="All Threads")
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_ticks([1, 5, 10, 50, 100, 500, 1000])
    axes[0].yaxis.set_ticklabels([1, 5, 10, 50, 100, 500, 1000])
    axes[0].set_ylabel("Number of comments")
    commenting_author_project_r.apply(Series).plot(
        kind='box', ax=axes[1], grid=False, logy=True, sym='.',
        rot=90, return_type='axes', color='steelblue',
        title="Research Threads")
    commenting_author_project_d.apply(Series).plot(
        kind='box', ax=axes[2], grid=False, logy=True, sym='.',
        rot=90, return_type='axes', color='steelblue',
        title="Discussion Threads")
    axes[1].yaxis.set_ticklabels([])
    axes[2].yaxis.set_ticklabels([])
    for i in range(3):
        axes[i].xaxis.set_ticks_position('bottom')
        axes[i].yaxis.set_ticks_position('left')
        # axes[i].set_ylim([0,1000])
    plt.savefig("FIGS/overview_box.png")
    # resetting mpl to values picked by seaborn.set
    mpl.rc("lines", markeredgewidth=0, solid_capstyle="round")


def plot_community_evolution(pm_frame, project, thread_type):
    """Area_plot of current, joined and left per project or thread.
    thread_type is 'all threads', 'research threads', or 'discussion
    threads'"""
    if not thread_type:
        logging.warning("Need explicit thread type")
        return
    try:
        int(project.split()[-1])
        as_threads = True
    except ValueError:
        as_threads = False
    if as_threads:
        data = pm_frame[['basic', thread_type]].loc[project].dropna()
        added, removed = _added_removed(data, thread_type, 'authors')
        size = get_len_v(data[thread_type, 'authors']) - added
        df = DataFrame({'joined': added, 'left': removed, 'current': size},
                       columns=["joined", "current", "left"], index=data.index)
        df.index = range(len(df))
        try:
            assert np.all(df == df.dropna())
        except AssertionError:
            logging.warning("Some nan-values still present")
        df.index.name = "Threads"
    else:
        data, positions = get_last(pm_frame, thread_type)
        added, removed = _added_removed(
            data, thread_type, 'authors (accumulated)')
        size = get_len_v(
            data[thread_type, 'authors (accumulated)'].dropna()) - added
        df = DataFrame({'joined': added, 'left': removed, 'current': size},
                       columns=["joined", "current", "left"])
        df.index = range(1, len(positions) + 1)
    mpl.style.use(SBSTYLE)
    axes = df.plot(kind="area", title="Community Evolution in {} ({})".format(
        project, thread_type),
        color=['seagreen', 'lightgrey', 'indianred'], stacked=True)
    axes.set_xticks(df.index)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    if as_threads:
        axes.set_xticklabels(data['basic', 'title'], rotation=90,
                             fontsize='small')
    else:
        xlabels = data.index.droplevel(1)
        axes.set_xticklabels(xlabels, rotation=90, fontsize='small')
    axes.set_ylabel('number of active commenters')


def project_participation_evolution(
        pm_frame, all_authors, n=2, skip_anon=True, research_only=False):
    """Assembles data on participation to projects with n as thresh.
    Returns DataFrame, index, selection and title for data for use
    by stacked bar-plot and heatmap functions."""
    if not research_only:
        thread_type = 'all threads'
        data, _ = get_last(pm_frame, thread_type)
        all_authors = list(all_authors)
        title = "Participation per project in Polymath\
                 (threshold = {})".format(n)
    else:
        thread_type = 'research threads'
        data, _ = get_last(pm_frame, thread_type)
        all_authors = set().union(
            *data['research threads', 'authors (accumulated)'])
        title = "Participation per project in Polymath\
                 (threshold = {}, only research-threads)".format(n)
    data.index = data.index.droplevel(1)
    author_project = DataFrame(columns=all_authors)
    for author in author_project.columns:
        author_project[author] = data[
            thread_type, 'authors (accumulated)'].apply(
                lambda project, author=author: author in project).astype('int')
    author_project = author_project.T
    author_project = author_project.sort_values(by=data.index.tolist(),
                                                ascending=False)
    author_project = author_project.drop(
        "Anonymous") if skip_anon else author_project
    select = author_project.sum(axis=1) >= n
    return author_project, data.index, select, title


def thread_participation_evolution(
        pm_frame, project, n=2, skip_anon=True, research_only=False):
    """Assembles data on participation to threads in project with n as thresh.
    Returns DataFrame, index, selection and title for data for use
    by stacked bar-plot and heatmap functions."""
    if not research_only:
        thread_type = 'all threads'
        title = "Participation per thread in {} (threshold = {})".format(
            project, n)
    else:
        thread_type = 'research threads'
        title = "Participation per thread in {}\
                 (threshold = {}, only research-threads)".format(project, n)
    data = pm_frame.loc[project][['basic', thread_type]]
    data = data.dropna()
    all_authors = set().union(*data[thread_type, 'authors'])
    author_thread = DataFrame(columns=all_authors)
    for author in author_thread.columns:
        author_thread[author] = data[thread_type, 'authors'].apply(
            lambda thread, author=author: author in thread)
    author_thread = author_thread.T
    author_thread = author_thread.sort_values(by=data.index.tolist(),
                                              ascending=False)
    author_thread = author_thread.drop(
        "Anonymous") if skip_anon else author_thread
    author_thread.columns.name = "Threads"
    select = author_thread.sum(axis=1) >= n
    return author_thread, data.index, select, title


def plot_participation_evolution(
        author_project, indices, select, title, fontsize=6):
    """Takes data from project_participation_evolution or
    thread_participation_evolution and plots as stacked bar-plot"""
    mpl.style.use(SBSTYLE)
    # factor = 30 - len(indices) if len(indices) <= 30 else 40 - len(indices)
    colors = [plt.cm.tab20(i) for i in range(len(indices))]
    axes = author_project.loc[select].plot(
        kind="bar", stacked=True, color=colors,
        title=title, fontsize=fontsize)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')


def plot_thread_evolution(pm_frame, project,
                          compress=1, sel=None, sharex=True):
    """Plots combined bar-plot and area-plot of participation to
    threads in project"""
    # data for evolution
    sel = [] if sel is None else sel
    thread_type = 'all threads'
    data = pm_frame.loc[project]
    added, removed = _added_removed(data, thread_type, 'authors')
    size = data[thread_type, 'number of authors'] - added
    df1 = DataFrame({'joined': added, 'left': removed, 'current': size},
                    columns=["joined", "current", "left"], index=data.index)
    df1.index.name = ""
    # data for engagement
    engagement = {}
    for thread_type in ['all threads',
                        'research threads',
                        'discussion threads']:
        engagement[thread_type] = data[
            thread_type, 'number of authors'] / data[
                thread_type, 'number of comments']
    df2 = DataFrame(engagement, index=data.index)
    sizes = (data['all threads', 'number of comments'] / compress).tolist()
    df2.index.name = ""
    df2_max = df2['all threads'].max()
    all_mean = df2['all threads'].mean()
    res_mean = df2['research threads'].mean()
    disc_mean = df2['discussion threads'].mean()
    # setting up plot
    mpl.style.use('seaborn-talk')
    _, axes = plt.subplots(2, 1, figsize=(15, 6),
                           squeeze=False, sharex=sharex)
    plt.subplots_adjust(hspace=0.2)
    # plot bottom
    df1.plot(kind="area", ax=axes[1][0], title="",
             color=['seagreen', 'lightgrey', 'indianred'], stacked=True)
    axes[1][0].set_ylabel('active commenters')
    axes[1][0].legend(bbox_to_anchor=(1.14, 1))
    # plot top
    df2[['research threads', 'discussion threads']].plot(
        kind='bar', ax=axes[0][0], color=['lightsteelblue', 'steelblue'],
        title="Community engagement in {}".format(project))
    axes[0][0].set_ylabel('comments per participant')
    axes[0][0].set_yticklabels(
        [round(1 / i, 2) for i in axes[0][0].get_yticks()])
    axes[0][0].yaxis.get_major_ticks()[0].label1.set_visible(False)
    axes[0][0].get_xaxis().set_ticks([])
    axes[0][0].set_xlabel("")
    xlims = (0.5, len(df1.index) + .5)
    axes[0][0].set_xlim(xlims)
    axes[0][0].set_ylim(0, df2_max * 1.1)
    axes[0][0].legend(bbox_to_anchor=(1.23, 1))
    axes[0][0].lines.append(
        mpl.lines.Line2D(list(xlims), [all_mean, all_mean], linestyle='--',
                         linewidth=.5, color='Blue', zorder=1,
                         transform=axes[0][0].transData))
    axes[0][0].lines.append(
        mpl.lines.Line2D(list(xlims), [res_mean, res_mean], linestyle='-',
                         linewidth=.5, color='lightsteelblue', zorder=1,
                         transform=axes[0][0].transData))
    axes[0][0].lines.append(
        mpl.lines.Line2D(list(xlims), [disc_mean, disc_mean], linestyle='-',
                         linewidth=.5, color='steelblue', zorder=1,
                         transform=axes[0][0].transData))
    for container in axes[0][0].containers:
        for i, child in enumerate(container.get_children()):
            child.set_x(df2.index[i] - sizes[i] / 2)
            plt.setp(child, width=sizes[i])
    all_threads = data['all threads'][
        ['number of comments', 'number of authors']]
    for i in list(set(sel).intersection(data.index)):
        axes[0][0].text(
            i + .25,
            df2.loc[i, 'all threads'] + .01,
            "{} comments\n {} commenters".format(
                all_threads.loc[i, 'number of comments'],
                all_threads.loc[i, 'number of authors']),
            ha="center", va="bottom", fontsize='small')


def plot_scatter_author_activity_projects(pm_frame, all_authors):
    """Scatter-plot of number of number of projects participated
    over avg number of comments per project"""
    author_project_bool, _, select_1, *_ = project_participation_evolution(
        pm_frame, all_authors, n=1, research_only=True)
    project_participation = author_project_bool.sum(axis=1)
    authors_1 = sorted([author for author, bool in select_1.items() if bool])
    author_counts, * _ = general_heatmap(
        pm_frame, all_authors, authors=authors_1,
        thread_level=False, binary=False)
    author_counts_mod = author_counts.replace(0, np.NaN)
    comment_participation = author_counts_mod.mean()
    df = pd.concat([project_participation, comment_participation],
                   axis=1).dropna()
    df.columns = ["number of projects participated",
                  "avg comments per project participated"]
    # axes = plt.subplot()
    axes = sns.swarmplot(
        x='number of projects participated',
        y='avg comments per project participated',
        order=range(1, 13),
        palette="muted",
        data=df)
    axes.set_xticks(range(10))
    axes.set_xlim([-.5, 9.5])
    axes.set_yticks(range(0, 700, 50))
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    for name in ['Timothy Gowers', 'Terence Tao', 'Gil Kalai']:
        x_val, y_val = df.loc[name].values
        axes.annotate(name.split()[1], xy=(x_val-1, y_val),
                  xytext=(x_val-2, y_val+20),
                  arrowprops=dict(facecolor='steelblue', shrink=0.05))
    e = mpl.patches.Ellipse(xy=(0, 368), width=.6, height=420, angle=0)
    e.set_alpha(.1)
    e.set_facecolor('gray')
    axes.add_artist(e)
    axes.annotate('Polymath 8', xy=(1, 400), xytext=(.5, 390))
    axes.set_title('Polymath participation and commenting activity')
    return df
    # df.plot(kind='scatter',
    #         x='number of projects participated',
    #         y='avg comments per project participated',
    #         color='lightsteelblue', ax=axes,
    #         title="Polymath participation and commenting activity")


def plot_scatter_author_activity_threads(pm_frame, all_authors):
    """Scatter-plot of number of number of threads participated
    over avg number of comments per thread"""
    thread_data, *_ = general_heatmap(
        pm_frame, all_authors, authors=None, binary=False,
        thread_level=True, binary_method='average', method='ward')
    thread_data = thread_data.T
    thread_bool = thread_data != 0
    thread_bool_sum = thread_bool.sum(axis=1)
    thread_data_mean = thread_data.replace(0, np.NaN)
    thread_data_mean = thread_data_mean.mean(axis=1)
    df_threads = pd.concat([thread_bool_sum, thread_data_mean], axis=1)
    df_threads.columns = ["number of threads participated",
                          "avg comments per thread participated"]
    axes = plt.subplot()
    axes.set_xticks([1] + list(range(5, 100, 5)))
    axes.set_yticks([1] + list(range(5, 30, 5)))
    axes.set_xlim(-1, 85)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    for name in ['Timothy Gowers', 'Terence Tao', 'Gil Kalai']:
        x_val, y_val = df_threads.loc[name].values
        axes.annotate(name.split()[1], xy=(x_val, y_val),
                  xytext=(x_val-7, y_val-2),
                  arrowprops=dict(facecolor='steelblue', shrink=0.05))
    df_threads.plot(kind='scatter',
                    x='number of threads participated',
                    y='avg comments per thread participated',
                    ax=axes, color='lightsteelblue',
                    title="Polymath participation and commenting activity")
    return df_threads


def scatter_author_profile(pm_frame, participant, thread_type="all threads",
                           x_measure=("i_graph", "eigenvector centrality"),
                           y_measure=("c_graph", "eigenvector centrality")):
    data, index = [], []
    for project, row in get_last(pm_frame,
                                 thread_type)[0][thread_type].iterrows():
        try:
            com_data = row['network'].author_frame.loc[participant]
        except KeyError:
            continue
        else:
            index.append(project[0])
        comments = com_data.loc['total comments']
        avg_counts = com_data.loc['word counts'] / comments
        network = row['network']
        x_graph = getattr(network, x_measure[0])
        y_graph = getattr(network, y_measure[0])
        x_data = network.centr_measures[x_measure[1]](x_graph)[participant]
        y_data = network.centr_measures[y_measure[1]](y_graph)[participant]
        data.append([comments, avg_counts, x_data, y_data])
    part_frame = DataFrame(data, index=index,
                           columns=['comments', 'average wordcount',
                                    " ".join(x_measure), " ".join(y_measure)]
                           )
    cut = (part_frame['average wordcount'].max() -
           part_frame['average wordcount'].min()) / 2
    axes = part_frame.plot(kind='scatter',
                           x=part_frame.columns[2], y=part_frame.columns[3],
                           c='average wordcount', s=part_frame['comments'],
                           cmap='viridis_r',
                           sharex=False)
    fake_legend([10, 150, 700], "Number of Comments")
    for project, values in part_frame.iterrows():
        if (values['average wordcount'] < cut) or (values['comments'] < 50):
            color = "darkgray"
        else:
            color = "lightgray"
        axes.text(values.iloc[2], values.iloc[3], project.split()[-1],
                  verticalalignment='center', horizontalalignment='center',
                  color=color)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_title("Activity and centrality of {} in all projects".format(
        participant))


#def scatter_projects(pm_frame, thread_type='all threads', network_type='i_graph'):
#    data = get_last(pm_frame, thread_type)[0][thread_type, 'network']
#    data = Series([getattr(netw, network_type) for network in data])
#    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
