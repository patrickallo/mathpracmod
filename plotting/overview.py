"""Module with plotting-functions to generate overview-plots
based on a pm_frame with all data from the PM-projects"""
# imports
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from author_network import SETTINGS
from notebook_helper.access_funs import get_last

SBSTYLE = SETTINGS['style']


def _added_removed(data, thread_type, authors):
    """helper-fun called by plot_community_evolution and
    plot_thread_evolution to compute differences between
    sets of participants in a data-frame"""
    added = (data[thread_type, authors] -
             data[thread_type, authors].shift(1)).apply(
                 lambda x: 0 if isinstance(x, float) else len(x))
    removed = (data[thread_type, authors].shift(1) -
               data[thread_type, authors]).apply(
                   lambda x: 0 if isinstance(x, float) else - len(x))
    return added, removed


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
        size = data[thread_type, 'authors'].apply(len) - added
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
        size = data[thread_type, 'authors (accumulated)'].dropna().apply(
            len) - added
        df = DataFrame({'joined': added, 'left': removed, 'current': size},
                       columns=["joined", "current", "left"])
        df.index = range(1, len(positions) + 1)
    mpl.style.use(SBSTYLE)
    axes = df.plot(kind="area", title="Community Evolution in {} ({})".format(
        project, thread_type),
        color=['sage', 'lightgrey', 'indianred'], stacked=True)
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
                lambda project, author=author: author in project)
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
    factor = 30 - len(indices) if len(indices) <= 30 else 40 - len(indices)
    colors = [plt.cm.Set1(factor * i) for i in range(len(indices))]
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
             color=['sage', 'lightgrey', 'indianred'], stacked=True)
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
