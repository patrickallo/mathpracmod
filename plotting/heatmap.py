"""Module with plotting-functions to generate heatmaps
based on a pm_frame with all data from the PM-projects"""
# imports

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pandas import DataFrame
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist

from author_network import SETTINGS
from notebook_helper.access_funs import get_last

SBSTYLE = SETTINGS['style']


def general_heatmap(pm_frame, all_authors,
                    authors=None, binary=False,
                    thread_type='all threads', thread_level=True,
                    cluster_authors=True, cluster_projects=True,
                    binary_method='average', method='ward', scale_data=False):
    if thread_level:
        authors_filtered = list(all_authors)
        try:
            authors_filtered.remove("Anonymous")
        except ValueError:
            pass
        data = pm_frame[thread_type, 'comment_counter']
    else:
        authors_filtered = list(all_authors) if not authors else authors
        try:
            authors_filtered.remove("Anonymous")
        except ValueError:
            pass
        data = get_last(pm_frame, thread_type)[0][thread_type]
        totals = data['number of comments (accumulated)'].values.reshape(-1, 1)
        data = data['comment_counter (accumulated)']
    if binary:
        as_matrix = np.array(
            [[True if author in data[thread] else False
              for author in authors_filtered]
             for thread in data.index])
        Z_author = linkage(as_matrix.T, method=binary_method, metric='hamming')
        if cluster_projects:
            Z_thread = linkage(as_matrix,
                               method=binary_method, metric='hamming')
        try:
            c, _ = cophenet(Z_author, pdist(as_matrix.T))
        except ValueError:
            pass
        else:
            logging.info("Cophenetic Correlation Coefficient with %s: %s",
                         binary_method, c)
    else:
        for_matrix = []
        for thread in data.index:
            new_row = [data.loc[thread][author] for author in authors_filtered]
            for_matrix.append(new_row)
        as_matrix = np.array(for_matrix)
        if scale_data:
            as_matrix = as_matrix / totals * 100
        if cluster_authors:
            Z_author = linkage(as_matrix.T, method=method, metric='euclidean')
            try:
                c, _ = cophenet(Z_author, pdist(as_matrix.T))
            except ValueError:
                pass
            else:
                logging.info("Cophenetic Correlation Coefficient with %s: %s",
                             method, c)
        if cluster_projects:
            Z_thread = linkage(as_matrix, method=method, metric='euclidean')
    # compute dendrogram and organise DataFrame
    df = DataFrame(as_matrix, columns=authors_filtered)
    if cluster_authors:
        ddata_author = dendrogram(Z_author, color_threshold=10,
                                  no_plot=True)
        cols = [authors_filtered[i] for i in ddata_author['leaves']]
        df = df[cols]
    if cluster_projects:
        ddata_thread = dendrogram(Z_thread, color_threshold=.07, no_plot=True)
        rows = [df.index[i] for i in ddata_thread['leaves']]
        df = df.reindex(rows)
    title = "Project-Engagement in Polymath ({})".format(thread_type)
    return df, binary, title


def plot_heatmap(df, binary, title, log=True, fontsize=8,
                 equal=True, figsize=None):
    # start setting up plots
    mpl.style.use(SBSTYLE)
    _, ax_heatmap = plt.subplots(figsize=figsize)
    my_lognorm = mpl.colors.LogNorm(vmax=df.values.max())
    # plot heatmap
    heatmap = ax_heatmap.pcolor(
        df, edgecolors='w',
        cmap=mpl.cm.binary if binary else mpl.cm.viridis_r,
        norm=my_lognorm if log else None)
    ax_heatmap.autoscale(tight=True)
    if equal:
        ax_heatmap.set_aspect('equal')
    ax_heatmap.xaxis.set_ticks_position('bottom')
    ax_heatmap.tick_params(bottom='off', top='off', left='off', right='off')
    ax_heatmap.set_title(title)
    ax_heatmap.set_yticks(np.arange(0.5, len(df.index) + .5, 1))
    ax_heatmap.set_yticklabels(df.index + 1, fontsize=fontsize)
    ax_heatmap.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax_heatmap.set_xticklabels(df.columns, rotation=90, fontsize=fontsize)
    if not binary:
        divider_h = make_axes_locatable(ax_heatmap)
        cax = divider_h.append_axes("right", "3%", pad="1%")
        un_vals = np.unique(df.values)
        plt.colorbar(heatmap, cax=cax, ticks=un_vals[::len(un_vals) // 5])
        cax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, pos: ('{:.2f} %'.format(my_lognorm.inverse(y)))))
    lines = (ax_heatmap.xaxis.get_ticklines() +
             ax_heatmap.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.tight_layout()


def project_heatmap(pm_frame,
                    project, binary=False, thread_type='all threads',
                    cluster_threads=True,
                    method='ward', binary_method='average',
                    skip_anon=True, log=False, fontsize=8):
    """
    Plots clustered heatmaps of thread-participation per author.
    Based on: https://gist.github.com/s-boardman/cef9675329a951e89e93
              https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    """
    data = pm_frame[thread_type].loc[project].copy().dropna()
    all_authors = sorted(list(data.iloc[-1]['authors (accumulated)']))
    if skip_anon:
        try:
            all_authors.remove("Anonymous")
        except ValueError:
            pass
    data = data['comment_counter']
    if binary:
        as_matrix = np.array([[True if author in data[thread] else False for
                               author in all_authors]
                              for thread in data.index])
        Z_author = linkage(as_matrix.T, method=binary_method, metric='hamming')
        if cluster_threads:
            Z_thread = linkage(
                as_matrix, method=binary_method, metric='hamming')
        try:
            c, _ = cophenet(Z_author, pdist(as_matrix.T))
            #print("Cophenetic Correlation Coefficient with {}: {}".format(
            #   binary_method, c))
        except ValueError:
            pass
    else:
        for_matrix = []
        for thread in data.index:
            new_row = [data.loc[thread][author] for author in all_authors]
            for_matrix.append(new_row)
        as_matrix = np.array(for_matrix)
        Z_author = linkage(as_matrix.T, method=method, metric='euclidean')
        if cluster_threads:
            Z_thread = linkage(as_matrix, method=method, metric='euclidean')
        try:
            c, _ = cophenet(Z_author, pdist(as_matrix.T))
            #print("Cophenetic Correlation Coefficient with {}: {}".format(
            #   method, c))
        except ValueError:
            pass
    ddata_author = dendrogram(Z_author, color_threshold=20, no_plot=True)
    df = DataFrame(as_matrix, columns=all_authors)
    cols = [all_authors[i] for i in ddata_author['leaves']]
    df = df[cols]
    if cluster_threads:
        ddata_thread = dendrogram(Z_thread, color_threshold=.07, no_plot=True)
        rows = [df.index[i] for i in ddata_thread['leaves']]
        df = df.reindex(rows)
    pm_frame.style.use('seaborn-poster')
    _, ax = plt.subplots(1, 1)
    my_lognorm = mpl.colors.LogNorm()
    heatmap = ax.pcolor(df,
                        edgecolors='k' if binary else 'w',
                        cmap=mpl.cm.binary if binary else mpl.cm.viridis_r,
                        norm=my_lognorm if log else None)
    ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
    ax.set_aspect('equal')  # ensure heatmap cells are square
    ax.xaxis.set_ticks_position('bottom')  # put column labels at the bottom
    ax.tick_params(bottom='off', top='off', left='off', right='off')
    ax.set_title("Thread-participation in {}".format(project))
    plt.yticks(np.arange(len(df.index)) + 0.5, df.index, fontsize=fontsize)
    plt.xticks(np.arange(len(df.columns)) + 0.5, df.columns,
               rotation=90, fontsize=fontsize)
    if not binary:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "3%", pad="1%")
        plt.colorbar(heatmap, cax=cax, ticks=[1, 2, 4, 10, 20, 40, 67])
        cax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, pos: ('{:.0f}'.format(my_lognorm.inverse(y)))))
    plt.tight_layout()
