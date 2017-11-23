"""
Module that includes the AuthorNetwork class,
which has a weighted nx.DiGraph based on a multi_comment_thread,
and a pandas.DataFrame with authors as index.
"""
# Imports
from bisect import insort
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime
from itertools import combinations, permutations
import logging
from operator import methodcaller
from textwrap import wrap
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame, date_range, Series
from pylab import ion

import access_classes as ac
import comment_thread as ct
import export_classes as ec

# Loading settings
SETTINGS, CMAP, _ = ac.load_settings()

# actions to be used as argument for --more
ACTIONS = {
    #  "network": "draw_graph",
    "author_activity": "plot_author_activity_bar",
    "author_activity_degree": "plot_activity_degree",
    "author_activity_prop": "plot_activity_prop",
    "centrality_measures": "plot_centrality_measures",
    "histogram": "plot_author_activity_hist",
    "discussion_centre": "draw_centre_discussion",
    "trajectories": "plot_i_trajectories",
    "distances": "plot_centre_dist",
    "delays": "plot_centre_closeness",
    "crowd": "plot_centre_crowd",
    "scatter": "scatter_authors",
    "hits": "scatter_authors_hits",
    "replies": "scatter_comments_replies"}


# Main
def main(project, **kwargs):
    """
    Creates AuthorNetwork (first calls CommentThread) based on supplied
    project, and optionally calls a method of AuthorNetwork.
    """
    do_more = kwargs.get('do_more', False)
    use_cached = kwargs.get('use_cached', False)
    cache_it = kwargs.get('cache_it', False)
    delete_all = kwargs.get('delete_all', False)
    try:
        an_mthread = ct.main(project, do_more=False,
                             use_cached=use_cached, cache_it=cache_it,
                             delete_all=delete_all)
    except AttributeError as err:
        logging.error("Could not create mthread: %s", err)
        sys.exit(1)
    a_network = AuthorNetwork(an_mthread)
    if do_more:
        the_project = project.replace(
            "pm", "Polymath ") if project.startswith(
                "pm") else project.replace("mini_pm", "Mini-Polymath ")
        do_this = methodcaller(ACTIONS[do_more], project=the_project)
        do_this(a_network)
        logging.info("Processing complete at %s", datetime.now())
    else:
        return a_network


# Classes
class AuthorFrame(DataFrame):
    """
    Subclass of DataFrame with authors as index
    """

    def __init__(self, author_color):
        super().__init__(
            {'color': list(author_color.values())},
            index=list(author_color.keys()))
        self.sort_index(inplace=True)
        self['word counts'] = np.zeros(self.shape[0])
        for i in range(1, 12):
            self["level {}".format(i)] = np.zeros(self.shape[0])


class AuthorInteractionGraph(nx.DiGraph):
    """
    Subclass of DiGraph with authors as nodes,
    and edges based on direct replies.
    """

    def __init__(self, author_names, thread_graph, no_loops):
        super().__init__()
        self.add_nodes_from(author_names)
        self.__i_graph_edges(thread_graph, no_loops)
        self = ac.scale_weights(self, "weight", "scaled_weight")

    def __i_graph_edges(self, thread_graph, no_loops):
        """Adds edges to interaction-graph."""
        for (source, dest) in thread_graph.edges():
            if no_loops and source == dest:
                continue
            source = thread_graph.node[source]
            source_author = source['com_author']
            source_time = source['com_timestamp']
            dest = thread_graph.node[dest]
            dest_author = dest['com_author']
            try:
                assert source_time >= dest['com_timestamp']
            except AssertionError:
                logging.warning("%s is not after %s", source, dest)
            try:
                edge = self[source_author][dest_author]
            except (AttributeError, KeyError):
                self.add_edge(
                    source_author, dest_author,
                    weight=1, timestamps=[source_time])
            else:
                edge['weight'] += 1
                insort(edge['timestamps'], source_time)
                assert edge['weight'] == len(edge['timestamps'])
        for _, _, data in self.edges(data=True):
            data['log_weight'] = np.log2(data['weight'])


class AuthorClusterGraph(nx.Graph):
    """
    Subclass of Graph with authors as nodes, and
    edges based on comments within episodes.
    """

    def __init__(self, author_names, author_episodes):
        super().__init__()
        self.add_nodes_from(author_names)
        self.__c_graph_edges(author_episodes)
        self = ac.scale_weights(self, "weight", "scaled_weight")

    def __c_graph_edges(self, author_episodes):
        """Adds edges to cluster-based graph"""
        for source_author, dest_author in combinations(
                author_episodes.keys(), 2):
            source_a_ep = {(thread, cluster): (weight, a_weight) for
                           (thread, cluster, weight, a_weight) in
                           list(author_episodes[source_author])}
            dest_a_ep = {(thread, cluster): (weight, a_weight) for
                         (thread, cluster, weight, a_weight) in
                         list(author_episodes[dest_author])}
            overlap = source_a_ep.keys() & dest_a_ep.keys()
            overlap = [(thread, cluster) for (thread, cluster) in list(
                overlap) if cluster is not None]
            if overlap:
                source_w = [source_a_ep[key] for key in overlap]
                dest_w = [dest_a_ep[key] for key in overlap]
                source_w, source_aw = zip(*source_w)
                dest_w, dest_aw = zip(*dest_w)
                weight = sum(source_w)
                assert weight == sum(dest_w)
                a_weight = np.minimum(
                    np.array(source_aw), np.array(dest_aw)).sum()
                self.add_edge(source_author,
                              dest_author,
                              weight=weight,
                              an_author_weight=a_weight,
                              log_weight=np.log2(weight),
                              simple_weight=len(overlap))


class AuthorClusterDiGraph(nx.DiGraph):
    """Subclass of DiGraph with authors as nodes, and edges
    based on comments within episodes"""

    def __init__(self, author_names, author_episodes):
        super().__init__()
        self.add_nodes_from(author_names)
        self.__c_graph_directed_edges(author_episodes)
        self = ac.scale_weights(self, "weight", "scaled_weight")

    def __c_graph_directed_edges(self, author_episodes):
        """Adds directed edges to cluster-based graph.
        Edge from a->b with weight w means that sum of all comments
        made by a in cluster in which b participated as well is w."""
        for source_author, dest_author in permutations(
                author_episodes.keys(), 2):
            source_a_ep = {(thread, cluster): a_weight for
                           (thread, cluster, _, a_weight) in
                           list(author_episodes[source_author])}
            dest_a_ep = [(thread, cluster) for
                         (thread, cluster, _, _) in
                         list(author_episodes[dest_author])]
            overlap = source_a_ep.keys() & dest_a_ep
            overlap = [(thread, cluster) for (thread, cluster) in list(
                overlap) if cluster is not None]
            if overlap:
                source_w = [source_a_ep[key] for key in overlap]
                weight = sum(source_w)
                self.add_edge(source_author,
                              dest_author,
                              weight=weight)


class AuthorNetwork(ec.GraphExportMixin, object):

    """
    Creates weighted nx.DiGraph of comments between authors,
    and stores author-info in DataFrame.
    Supplies several methods for plotting aspects of author-activity.

    Attributes:
        mthread: instance of ct.MultiCommentThread.
        all_thread_graphs: DiGraph of ct.MultiCommentThread.
        node_name: dict with nodes as keys and authors as values.
        positions: None or stored list of positions generated by spring
        author_frame: pandas.DataFrame with authors as index.
        i_graph: weighted nx.DiGraph (weighted directed edges between authors)
        c_graph: weighted ng.Graph (weighted undirected edges between authors)

    Methods:
        author_count: returns Series with
                      authors as index and num of comments as values
        __i_graph_edges, __c_graph_edges, __author_activity, __author_replies,
        __sort_timestamps, __check_author_frame: methods called by init
        plot_author_activity_bar: plots commenting activity in bar-chart
        plot_centrality_measures: plots bar-graph with centr-measures for
                                  each author
        plot_activity_degree: plots bar of number of comments and line of
                              degree-centrality
        plot_author_activity_pie: plots commenting activity in pie-chart
        plot_author_activity_hist: plots histogram of commenting activity
        weakly connected components: returns generator of
                                     weakly connected components
        draw_graph: draws DiGraph of author_network
        draw_centre_discussion: draws and modifies graph of who moves in/out
                                of centre of discussion

    """

    def __init__(self, an_mthread, no_loops=True):
        super(AuthorNetwork, self).__init__()
        # attributes from argument MultiCommentThread
        self.mthread = an_mthread
        self.all_thread_graphs = an_mthread.graph
        self.node_name = an_mthread.node_name
        self.positions = None  # positions stored when draw_graph is called
        # create author_frame with color as column
        self.author_frame = AuthorFrame(an_mthread.author_color)
        # Initialize and populate DiGraph with authors as nodes.
        self.i_graph = AuthorInteractionGraph(self.author_frame.index,
                                              self.all_thread_graphs,
                                              no_loops=no_loops)
        author_nodes, author_episodes = self.__author_activity()
        # create column in author_frame from author_nodes
        self.author_frame["comments"] = Series(
            {key: sorted(value) for (key, value) in author_nodes.items()})
        # create column in author_frame from author_episodes
        self.author_frame["episodes"] = Series(author_episodes)
        self.c_graph = AuthorClusterGraph(self.author_frame.index,
                                          author_episodes)
        self.c_dgraph = AuthorClusterDiGraph(self.author_frame.index,
                                             author_episodes)
        # removed unused levels-columns in author_frame
        self.author_frame = self.author_frame.loc[
            :, (self.author_frame != 0).any(axis=0)]
        # add columns with total comments and timestamps to author_frame
        self.author_frame['total comments'] = self.author_frame.iloc[
            :, 2:].sum(axis=1)
        self.author_frame['timestamps'] = [self.i_graph.node[an_author][
            "post_timestamps"] for an_author in self.author_frame.index]
        self.__check_author_frame()
        # adding first and last comment to author_frame
        self.author_frame['first'] = ac.get_first_v(
            self.author_frame['timestamps'])
        self.author_frame['last'] = ac.get_last_v(
            self.author_frame['timestamps'])
        # generate random angles for each author (to be used in
        # draw_centre_discussion)
        self.author_frame['angle'] = np.linspace(
            0, 360, len(self.author_frame), endpoint=False)
        self.__author_replies()
        # adding multiple centrality-measures to author-frame
        # ToDo: see if computing the values can be delayed until the
        # measures are needed...
        self.centr_measures = OrderedDict([
            ('degree centrality', nx.degree_centrality),
            ('eigenvector centrality', nx.eigenvector_centrality),
            ('betweenness centrality', nx.betweenness_centrality),
            ('closeness centrality', nx.closeness_centrality),
            ('Katz centrality', nx.katz_centrality),
            ('page rank', nx.pagerank),
            ('in-degree', nx.in_degree_centrality),
            ('out-degree', nx.out_degree_centrality)])
        self.g_types = ["interaction", "cluster", "directed cluster"]
        # self.__add_centr_measures() # this should go as well!

    def author_count(self):
        """Returns series with count of authors (num of comments per author)"""
        return self.author_frame['total comments']

    def __author_activity(self):
        """Iterates over mthread to collect author-info,
        adds info to data-frame, timestamps to network,
        and returns dicts with nodes and episodes
        linked to each author"""
        author_nodes = defaultdict(list)
        author_episodes = defaultdict(set)
        for node, data in self.all_thread_graphs.nodes(data=True):
            # set comment_levels in author_frame, and
            # set data for first and last comment in self.graph
            the_author = data['com_author']
            the_level = 'level {}'.format(data['com_depth'])
            the_date = data['com_timestamp']
            the_count = len(data['com_tokens'])
            the_thread, the_cluster = data['com_thread'], data['cluster_id']
            self.author_frame.ix[the_author, the_level] += 1
            self.author_frame.ix[the_author, 'word counts'] += the_count
            author_nodes[the_author].append(node)
            author_episodes[the_author].add((
                the_thread.path.split('/')[-2],
                *the_cluster))
            # adding timestamp or creating initial list of timestamps for
            # auth in DiGraph
            try:
                insort(self.i_graph.node[the_author]['post_timestamps'],
                       the_date)
            except KeyError:
                self.i_graph.node[the_author]['post_timestamps'] = [the_date]
        return author_nodes, author_episodes

    def __author_replies(self):
        """Iterates over authors to create replies columns in author_frame"""
        for author in self.author_frame.index:
            for label in ['replies (all)',
                          'replies (direct)',
                          'replies (own excl.)']:
                self.author_frame.ix[author, label] = sum(
                    [self.mthread.comment_report(i)[label]
                     for i in self.author_frame.ix[author, "comments"]])

    def __check_author_frame(self):
        try:
            assert (ac.get_len_v(self.author_frame['comments']) ==
                    self.author_frame['total comments']).all()
            assert (ac.get_len_v(self.author_frame['timestamps']) ==
                    self.author_frame['total comments']).all()
        except AssertionError as err:
            logging.error("Numbers of comments for %s do not add up: %s",
                          list(self.mthread.thread_url_title.values()), err)

    def __get_author_activity_bylevel(self):
        cols = self.author_frame.columns[
            self.author_frame.columns.str.startswith('level')]
        levels = self.author_frame[cols].sort_values(
            cols.tolist(), ascending=False)
        colors = [plt.cm.Vega10(i) for i in range(len(levels))]
        return levels, colors

    def __get_centrality_measures(self,
                                  g_type, **kwargs):
        """ Helper-fun to get centr-measures.
        Arguments:
            g_type: graph-type,
                    'cluster' or 'directed cluster' or 'interaction'
            **kwargs: measures (list or None),
                      weight (string or None),
                      sort (bool).
        Returns: [TODO: see if the col-names can be dropped]
            a DataFrame with the measures
            a dict of the means"""
        measures = kwargs.pop('measures', None)
        weight = kwargs.pop('weight', None)
        sort = kwargs.pop('sort', True)
        to_undirected = kwargs.pop("to_undirected", False)
        if g_type not in self.g_types:
            raise ValueError
        if measures:
            measures_dict = OrderedDict()
            for measure in measures:
                measures_dict[measure] = self.centr_measures[measure]
        else:
            measures_dict = self.centr_measures.copy()
        if not set(measures_dict).issubset(self.centr_measures.keys()):
            raise ValueError
        if g_type == "cluster":
            graph = self.c_graph
            logging.info("In/out-degree are removed (if present)\
                for undirected graphs")
            measures_dict.pop('in-degree', None)
            measures_dict.pop('out-degree', None)
        elif g_type == "directed cluster":
            graph = self.c_dgraph.to_undirected() if to_undirected\
                else self.c_dgraph
        else:  # consider additional option for randomized graph
            graph = self.i_graph.to_undirected() if to_undirected\
                else self.i_graph
        centrality = DataFrame(index=self.author_frame.index)
        for measure, fun in measures_dict.items():
            try:
                centrality[measure] = Series(fun(graph, weight=weight))
            except TypeError:
                centrality[measure] = Series(fun(graph))
            except ZeroDivisionError as err:
                logging.warning("error with %s: %s", measure, err)
                centrality[measure] = Series(
                    np.zeros_like(measure.index))
            except nx.PowerIterationFailedConvergence as err:
                logging.info("error with %s: %s", measure, err)
        centrality.columns = [g_type + " " + measure for measure in
                              centrality.columns]
        if sort:
            centrality = centrality.sort_values(
                centrality.columns[0], ascending=False)
        return centrality

    def __get_centre_distances(self, thresh, split=False):
        """Helper-function to create df of distances from centre
        of discussion (time since last comment in days).
        Returns df and indices of low/high commenters."""
        timestamps = self.author_frame['timestamps'].copy()
        sizes = ac.get_len_v(timestamps)
        # splitting authors with more/less than avg of timestamps
        authors_high = timestamps[sizes >= sizes.mean()].index
        authors_low = timestamps[
            (sizes >= thresh) & (sizes < sizes.mean())].index
        # filter out contributors with ≤ thresh comments
        timestamps = timestamps[sizes >= thresh]
        # create new index based on existing stamps + daily daterange
        index = pd.Index(
            np.sort(np.concatenate(timestamps))).unique()
        daily = pd.date_range(start=index[0], end=index[-1], freq='H')
        index = index.union(daily).unique()
        # create empty dataframe
        data = DataFrame(
            np.zeros((len(index), len(timestamps.index)), dtype='bool'),
            index=index, columns=timestamps.index)
        # filling in Boolean values for author-time pairs
        for name, stamps in timestamps.iteritems():
            data.loc[stamps, name] = np.ones_like(stamps, dtype='bool')
        # create array of intervals (in days) and add as col to df
        intervals = np.zeros(data.shape[0], dtype='float64')
        intervals[1:] = np.diff(data.index)
        data['intervals'] = (intervals * 1e-9) / (60**2 * 24)
        # adding intervals for each author to data
        for name in timestamps.index:
            name_intervals = name + "-intervals"
            # adding all potential intervals
            data[name_intervals] = data['intervals']
            # setting intervals to zero for all comments
            data.loc[data[name], name_intervals] = 0
            # create mask to filter out before first comment
            data[name] = data[name].cumsum()
            mask = data[name] == 0
            # compute actual distances from centre
            data[name] = data.groupby([name])[name_intervals].cumsum()
            # insert nan where appropriate based on mask
            data.loc[mask, name] = np.nan
        if split:
            out = data[authors_high], data[authors_low]
        else:
            out = data[timestamps.index]
        return out

    def __hits(self, g_type="interaction"):
        if g_type == "interaction":
            graph = self.i_graph
        elif g_type == "directed cluster":
            graph = self.c_dgraph
        else:
            raise ValueError
        hubs, authorities = nx.hits(graph)
        hits = DataFrame([hubs, authorities], index=['hubs', 'authorities']).T
        hits['word counts'] = self.author_frame['word counts']
        hits['total comments'] = self.author_frame['total comments']
        return hits

    def __show_threads(self, axes):
        """Helper function to show duration of threads as
        hline on supplied axes"""
        iterate_over = self.mthread.t_bounds.values()
        iterate_over = zip(
            iterate_over,
            np.linspace(
                axes.get_ylim()[1] / 10,
                axes.get_ylim()[1],
                num=len(iterate_over),
                endpoint=False))
        for (thread_start, thread_end), height in iterate_over:
            start = mdates.date2num(thread_start)
            stop = mdates.date2num(thread_end)
            axes.hlines(height,
                        start, stop, alpha=.3)

    def plot_author_activity_bar(self, what='by level', **kwargs):
        """Shows plot of number of comments / wordcount per author.
        what can be either 'by level' or 'word counts'"""
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        plt.style.use(SETTINGS['style'])
        if what == "by level":
            levels, colors = self.__get_author_activity_bylevel()
            total_num_of_comments = int(levels.sum().sum())
            axes = levels.plot(
                kind='barh', stacked=True, color=colors,
                title='Comments per author (total: {})'.format(
                    total_num_of_comments),
                fontsize=fontsize)
            axes.set_yticklabels(levels.index, fontsize=fontsize)
        elif what == "word counts":
            word_counts = self.author_frame[what].sort_values(ascending=False)
            total_word_count = int(word_counts.sum())
            axes = word_counts.plot(
                kind='bar', logy=True,
                title='Word-count per author in {} (total: {})'.format(
                    project, total_word_count),
                fontsize=fontsize)
            axes.xaxis.set_ticks_position('bottom')
            axes.yaxis.set_ticks_position('left')
        else:
            raise ValueError
        ac.show_or_save(show)

    def corr_centrality_measures(self, g_type='interaction', weight=None):
        """Returns DataFrame with standard Pearson-correlation between
        the different centrality-measures for chosen graph-type"""
        centrality = self.__get_centrality_measures(
            g_type, measures=self.centr_measures, weight=weight)
        correlation = centrality.corr()
        return correlation

    def plot_centrality_measures(self, **kwargs):
        """Shows plot of degree_centrality for each author
        (only if first measure is non-zero)
        kwargs:
            g_type: graph_type, defaults to 'interaction'
            measures: list of measures, defaults to all
            weight: string, defaults to None
            delete_on: col-index (int), defaults to None
            thresh: threshold for deleting (int), defaults to 0
            project, show, and fontsize"""
        g_type = kwargs.pop("g_type", "interaction")
        measures = kwargs.pop("measures", self.centr_measures.copy())
        weight = kwargs.pop("weight", None)
        delete_on = kwargs.pop("delete_on", None)
        thresh = kwargs.pop("thresh", 0)
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        centrality = self.__get_centrality_measures(
            g_type, measures=measures, weight=weight)
        centr_cols = centrality.columns
        means = centrality.mean().to_dict()
        if delete_on is not None:
            centrality = centrality[centrality[centr_cols[delete_on]] > thresh]
        print(measures)
        colors = ac.color_list(len(measures),
                               SETTINGS['vmin'], SETTINGS['vmax'],
                               factor=15)
        full_measure_names = centrality.columns
        centrality.columns = [
            col.replace(g_type + " ", "") for col in centrality.columns]
        plt.style.use(SETTINGS['style'])
        axes = centrality.plot(
            kind='bar', color=colors,
            title="Centrality-measures for {} ({}-graph)".format(
                project, g_type).title())
        for measure, color in zip(full_measure_names, colors):
            the_mean = means[measure]
            axes.lines.append(
                mlines.Line2D(
                    [-.5, len(centrality.index) - .5],
                    [the_mean, the_mean],
                    linestyle='-', linewidth=.5,
                    color=color, zorder=1,
                    transform=axes.transData))
        axes.set_xticklabels(centrality.index, fontsize=fontsize)
        ac.show_or_save(show)

    def plot_activity_degree(self, **kwargs):
        """Shows plot of number of comments (bar) and network-measures (line)
        for all authors with non-null centrality-measure
        kwargs:
            g_type: graph_type, defaults to 'interaction'
            measures: list of measures, defaults to all
            weight: string, defaults to None
            delete_on: col-index (int), defaults to None
            thresh: threshold for deleting (int), defaults to 0
            project, show, and fontsize"""
        g_type = kwargs.pop("g_type", "interaction")
        measures = kwargs.pop("measures", self.centr_measures)
        weight = kwargs.pop("weight", None)
        delete_on = kwargs.pop("delete_on", None)
        thresh = kwargs.pop("thresh", 0)
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        # data for centrality measures
        if measures == ['hits']:
            centr_cols = ['hubs', 'authorities']
            centrality = self.__hits()[centr_cols].sort_values(
                centr_cols[0], ascending=False)
        else:
            centrality = self.__get_centrality_measures(
                g_type, measures=measures, weight=weight)
            centr_cols = centrality.columns
        if delete_on is not None:
            centrality = centrality[centrality[centr_cols[delete_on]] > thresh]
        # data for commenting-activity (limited to index of centrality)
        comments, colors = self.__get_author_activity_bylevel()
        comments = comments.loc[centrality.index]
        plt.style.use(SETTINGS['style'])
        axes = comments.plot(
            kind='bar', stacked=True, color=colors,
            title="Commenting activity and {} for {}".format(
                ", ".join(measures), project).title(),
            fontsize=fontsize)
        axes.set_ylabel("Number of comments")
        axes.xaxis.set_ticks_position('bottom')
        axes2 = axes.twinx()
        axes2.set_ylabel("Measures")
        col_marker = list(zip(centr_cols, "oDsv^"))
        for col, marker in col_marker:
            axes2.plot(axes.get_xticks(), centrality[col].values,
                       linestyle=':', marker=marker, markersize=5,
                       linewidth=.7, color='darkgray')
        the_lines = [mlines.Line2D([], [], color='darkgray',
                                   linewidth=.7,
                                   marker=marker,
                                   markersize=5,
                                   label=col.replace(g_type + " ", ""))
                     for (col, marker) in col_marker]
        axes2.legend(handles=the_lines,
                     bbox_to_anchor=(.83, 1))
        ac.show_or_save(show)

    def plot_activity_prop(self, **kwargs):  # candidate for removal
        """Shows plot of number of comments (bar) and proportion
        level-1 / higher-level comment (line) for all authors"""
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        plt.style.use(SETTINGS['style'])
        cols = self.author_frame.columns[
            self.author_frame.columns.str.startswith('level')].tolist()
        data = self.author_frame[cols].copy()
        data['proportion'] = (data[cols[1:]].sum(axis=1) /
                              data[cols].sum(axis=1))
        colors = [plt.cm.Vega10(i) for i in range(len(data.index))]
        axes = data[cols].plot(
            kind='bar', stacked=True, color=colors,
            title="""Commenting activity and proportion of higher-level
                comments for {}""".format(project).title(),
            fontsize=fontsize)
        axes.set_ylabel("Number of comments")
        axes.legend(bbox_to_anchor=(0.165, 1))
        axes2 = axes.twinx()
        axes2.set_ylabel("Proportion of Higher-level comments")
        axes2.plot(axes.get_xticks(), data['proportion'].values,
                   linestyle=':', marker='.', markersize=10, linewidth=.7,
                   color='darkgrey')
        the_lines = [mlines.Line2D([], [], color='gray', linestyle=':',
                                   marker='.', markersize=10,
                                   label="Proportion")]
        axes2.legend(handles=the_lines,
                     bbox_to_anchor=(1, 1))
        ac.show_or_save(show)

    def plot_author_activity_pie(self, what='total comments', **kwargs):
        """Shows plot of commenting activity as piechart
           what can be either 'total comments' (default) or 'word counts'"""
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        if what not in set(['total comments', 'word counts']):
            raise ValueError
        comments = self.author_frame[[what, 'color']].sort_values(
            what, ascending=False)
        thresh = int(np.ceil(comments[what].sum() / 100))
        whatcounted = 'comments' if what == 'total comments' else 'words'
        comments.index = [[x if y >= thresh else "fewer than {} {}"
                           .format(thresh, whatcounted) for
                           (x, y) in comments[what].items()]]
        merged_commenters = comments.index.value_counts()[0]
        comments = DataFrame({
            'totals': comments[what].groupby(comments.index).sum(),
            'maxs': comments[what].groupby(comments.index).max(),
            'color': comments['color'].groupby(
                comments.index).max()}).sort_values(
                    'maxs', ascending=False)
        for_pie = comments['totals']
        for_pie.name = ""
        colors = ac.color_list(comments['color'],
                               SETTINGS['vmin'], SETTINGS['vmax'],
                               cmap=CMAP)
        plt.style.use(SETTINGS['style'])
        title = "Activity per author for {}".format(project).title()
        if what == "total comments":
            title += ' ({} comments, {} with fewer than {} comments)'.format(
                int(comments['totals'].sum()),
                merged_commenters,
                thresh)
        else:
            title += ' ({} words, {} with fewer than {} words)'.format(
                int(comments['totals'].sum()),
                merged_commenters,
                thresh)
        for_pie.plot(
            kind='pie', autopct='%.2f %%', figsize=(6, 6),
            labels=for_pie.index,
            colors=colors,
            title=('\n'.join(wrap(title, 60))),
            fontsize=fontsize)
        ac.show_or_save(show)

    def plot_author_activity_hist(self,
                                  what='total comments', bins=10,
                                  **kwargs):
        """Shows plot of histogram of commenting activity.
           What can be either 'total comments' (default) or 'word counts'"""
        project, show, _ = ac.handle_kwargs(**kwargs)
        if what not in set(['total comments', 'word counts']):
            raise ValueError
        comments = self.author_frame[what]
        plt.style.use(SETTINGS['style'])
        _, axes = plt.subplots()
        comments.hist(bins=bins, grid=False, ax=axes)
        axes.set_title("Histogram of {} for {}".format(what, project))
        axes.set_xlim(1)
        axes.set_yticks(axes.get_yticks()[1:])
        ac.show_or_save(show)

    def plot_edge_weight_distribution(self, **kwargs):
        """Shows plot of histogram (optionally kde) of edge_weights"""
        g_type = kwargs.pop("g_type", "interaction")
        weight = kwargs.pop("weight", "weight")
        transform = kwargs.pop("transform", lambda x: x)
        kind = kwargs.pop("kind", "hist")
        project, show, _ = ac.handle_kwargs(**kwargs)
        graph = self.i_graph if g_type == "interaction" else self.c_graph
        data = Series(
            [data[weight] for _, _, data in graph.edges(data=True)])
        data = transform(data)
        plt.style.use(SETTINGS['style'])
        _, axes = plt.subplots()
        data.plot(kind=kind, ax=axes)
        axes.set_title("Distribution of edge-weights in {} {}-network".format(
            project, g_type))
        ac.show_or_save(show)

    def scatter_authors(self, **kwargs):
        """Scatter-plot with position based on interaction and cluster
        measure, color based on number of comments, and size on avg comment
        length
        kwargs:
            measure: string, defaults to betweenness centrality
            weight: dict ofstrings,
                defaults to {"interaction": None, "cluster": None)
            thresh: threshold for showing labels, defaults to 15
            xlim, ylim: ints passed to axes.set_xlim/yLim
            project, show, and fontsize
        Weights can be chosen as follows:
        For interaction-network 'weight' is the number of replies, and
        'log_weight' the log2 of weight.
        For cluster-network 'simple_weight' is the number of common episodes,
        'weight' the sum of the sizes (number of comments) of the common
        episodes, and 'author_weight' the sum of the least engagements in each
        common episode (if we look at a and b in c, the author-weight for c
        is the minimum of a's comments in c and b's comments in c).
        """
        cluster_g_type = kwargs.pop("cluster_g_type", "directed cluster")
        measure = kwargs.pop("measure", "degree centrality")
        weight = kwargs.pop("weight", {'interaction': None,
                                       'cluster': None})
        to_undirected = kwargs.pop("to_undirected", False)
        thresh = kwargs.pop("thresh", 15)
        xlim, ylim = kwargs.pop("xlim", None), kwargs.pop("ylim", None)
        add_diagonal = kwargs.pop("add_diagonal", True)
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        x_measure, y_measure = [" ".join([netw, measure]) for netw in
                                ["interaction", cluster_g_type]]
        # assemble data
        data = self.author_frame[['total comments', 'word counts']].copy()
        data[x_measure] = self.__get_centrality_measures(
            "interaction", measures=[measure], weight=weight["interaction"],
            to_undirected=to_undirected)
        data[y_measure] = self.__get_centrality_measures(
            cluster_g_type, measures=[measure], weight=weight["cluster"],
            to_indirected=to_undirected)
        axes = data.plot(
            kind='scatter',
            x=x_measure, y=y_measure,
            c='total comments',
            s=data['word counts'] / data['total comments'],
            cmap="viridis_r",
            sharex=False,
            title="Author-activity and centrality in {}".format(project))
        max_val = data[[x_measure, y_measure]].max().max()
        if add_diagonal:
            axes.plot([0, max_val], [0, max_val], color="k", alpha=.5)
        if xlim:
            axes.set_xlim(xlim)
        if ylim:
            axes.set_ylim(ylim)
        for name, vals in data.iterrows():
            if vals['total comments'] >= thresh:
                axes.text(vals[x_measure], vals[y_measure], name,
                          fontsize=fontsize)

        ac.fake_legend([50, 100, 250], title="Average wordcount of comments")
        ac.show_or_save(show)

    def scatter_authors_hits(self, thresh=10, **kwargs):
        """Scatter-plot based on hits-algorithm for hubs and authorities"""
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        hits = self.__hits()
        axes = hits.plot(
            kind='scatter',
            x='hubs', y='authorities',
            c='total comments',
            s=hits['word counts'] / hits['total comments'],
            cmap="viridis_r",
            sharex=False,
            title="Hubs and Authorities in {}".format(project))

        for name, data in hits.iterrows():
            if data['total comments'] >= thresh:
                axes.text(data['hubs'], data['authorities'], name,
                          fontsize=fontsize)
        ac.fake_legend([50, 100, 250], title="Average wordcount of comments")
        ac.show_or_save(show)

    def scatter_comments_replies(self, **kwargs):
        """Scatter-plot of comments vs direct replies received"""
        project, show, _ = ac.handle_kwargs(**kwargs)
        data = self.author_frame[['total comments',
                                  'replies (direct)']]
        data.plot(
            kind='scatter',
            x="total comments", y='replies (direct)',
            sharex=False,
            title="total comments vs replies in {}".format(project))
        ac.show_or_save(show)

    def plot_i_trajectories(self, **kwargs):
        """Plots interaction-trajectories for each pair of contributors.
        kwargs:
            select: list selected diads (takes precedence over thresh)
            thresh: min of interactions for inclusion (int)
            l_trhesh: threshold for inclusion in legend, defaults to 5
            project, show"""
        remove_self_replies = kwargs.pop("no_loops", True)
        select = kwargs.pop("select", None)
        thresh = kwargs.pop("thresh", None)
        l_thresh = kwargs.pop("l_thresh", 5)
        project, show, _ = ac.handle_kwargs(**kwargs)
        if remove_self_replies:
            graph = self.i_graph.copy()  # defensive copy
            graph.remove_edges_from([(i, i) for i in graph.nodes()])
        else:
            graph = self.i_graph.copy()
        trajectories = {}
        for (source, dest, data) in graph.edges(data=True):
            name = " / ".join([source, dest])
            trajectories[name] = Series(Counter(data['timestamps']),
                                        name=name)
        try:
            tr_data = DataFrame(trajectories)
        except ValueError as err:
            print("Could not create DataFrame: ", err)
        else:
            tr_data = tr_data.fillna(0).cumsum().sort_index()
            col_order = tr_data.iloc[-1].sort_values(ascending=False).index
            tr_data = tr_data[col_order]
            title = "Interaction trajectories for {}".format(project)
            if select:
                tr_data = tr_data.iloc[:, :select]
                title += " ({} largest)".format(select)
            elif thresh:
                tr_data = tr_data.loc[:, ~(tr_data < thresh).all(axis=0)]
                title += " (minimally {} interactions)".format(thresh)
            plt.style.use(SETTINGS['style'])
            _, axes = plt.subplots()
            for col in col_order[:l_thresh]:
                tr_data[col].plot(ax=axes, label=col)
            for col in col_order[l_thresh:]:
                tr_data[col].plot(ax=axes, label=None)
            axes.legend(labels=col_order[:l_thresh], loc='best')
            axes.set_title("Interaction trajectories for {}".format(project))
            axes.xaxis.set_ticks_position('bottom')
            axes.yaxis.set_ticks_position('left')
            ac.show_or_save(show)

    def plot_centre_closeness(self, thresh=10, ylim=16, **kwargs):
        """Boxplot of time before return to centre for core authors"""
        project, show, _ = ac.handle_kwargs(**kwargs)
        timestamps = self.author_frame['timestamps'].apply(np.array)
        try:
            timestamps.drop("Anonymous", inplace=True)
        except ValueError:
            pass
        delays = timestamps.apply(np.diff)
        delays = delays[ac.get_len_v(delays) >= thresh]
        delays = delays.map(ac.to_days)
        plt.style.use(SETTINGS['style'])
        _, axes = plt.subplots()
        bplot = plt.boxplot(delays, sym=None,
                            showmeans=True, meanline=True)
        for key in ['whiskers', 'boxes', 'caps']:
            plt.setp(bplot[key], color='steelblue')
        plt.setp(bplot['means'], color="firebrick")
        axes.set_xticklabels(delays.index, rotation=40, ha='right')
        axes.set_xlabel("Participants with at least {} comments".format(
            thresh))
        axes.set_yticks(np.logspace(-1, 3, num=5, base=2))
        axes.set_ylabel("Delay in days")
        if ylim:
            axes.set_ylim(0, ylim)
        axes.set_title("Delays between comments in {}".format(project))
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        ac.show_or_save(show)

    def plot_centre_dist(self, thresh=2, show_threads=True, **kwargs):
        """Plots time elapsed since last comment for each participant"""
        project, show, _ = ac.handle_kwargs(**kwargs)
        data_high, data_low = self.__get_centre_distances(
            thresh, split=True)
        # set up and create plots
        plt.style.use(SETTINGS['style'])
        _, axes = plt.subplots()
        colors_high = ac.color_list(
            self.author_frame.loc[data_high.columns, 'color'],
            SETTINGS['vmin'], SETTINGS['vmax'],
            cmap=CMAP)
        colors_low = ac.color_list(
            self.author_frame.loc[data_low.columns, 'color'],
            SETTINGS['vmin'], SETTINGS['vmax'],
            cmap=CMAP)
        data_high.plot(ax=axes, color=colors_high, legend=False)
        data_low.plot(ax=axes, alpha=.1, color=colors_low, legend=False)
        axes.set_ylabel("Days elapsed since last comment")
        axes.set_title("Distance from centre of discussion\n{}".format(
            project))
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        axes.set_ylim(0, None)
        if show_threads:
            self.__show_threads(axes)
        ac.show_or_save(show)

    def plot_centre_crowd(self, thresh=2, show_threads=False, **kwargs):
        """Plotting evolution of number of participants close to centre"""
        project, show, _ = ac.handle_kwargs(**kwargs)
        data = self.__get_centre_distances(thresh, split=False)
        data_close = DataFrame({
            '3 hours': data[data <= .125].count(axis=1),
            '6 hours': data[(data <= .25) & (data > .125)].count(axis=1),
            '12 hours': data[(data <= .5) & (data > .25)].count(axis=1),
            '24 hours': data[(data <= 1) & (data > .5)].count(
                axis=1)},
                               columns=['3 hours', '6 hours',
                                        '12 hours', '24 hours'])
        plt.style.use(SETTINGS['style'])
        y_max = data_close.sum(axis=1).max()
        _, axes = plt.subplots()
        data_close.plot(kind="area", ax=axes, stacked=True,
                        color=['limegreen', 'darkslategray',
                               'steelblue', 'lightgray'])
        axes.set_yticks(range(1, y_max + 1))
        axes.set_ylabel("Number of participants")
        axes.set_title("Crowd close to the centre of discussion in {}".format(
            project))
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        if show_threads:
            self.__show_threads(axes)
        ac.show_or_save(show)

    def w_connected_components(self, graph_type):
        """Returns weakly connected components as generator of list of nodes.
        This ignores the direction of edges."""
        graph = self.c_graph if graph_type == "cluster" else self.i_graph
        return nx.weakly_connected_components(graph)

    def draw_centre_discussion(self,
                               regular_intervals=False,
                               skips=2, zoom=12, **kwargs):
        """Draws part of nx.DiGraph to picture who's
        at the centre of activity"""
        _, show, _ = ac.handle_kwargs(**kwargs)
        activity_df = self.author_frame[
            ['color', 'angle', 'timestamps']].copy()
        if not regular_intervals:
            intervals = np.concatenate(activity_df['timestamps'].values)
            intervals.sort(kind='mergesort')
            intervals = intervals[::skips]
        else:
            start = np.min(activity_df['timestamps'].apply(np.min))
            stop = np.max(activity_df['timestamps'].apply(np.max))
            intervals = date_range(start, stop)[::skips]
        x_max, y_max = 0, 0
        for interval in intervals:
            interval_data = activity_df['timestamps'].copy().apply(np.array)
            interval_data = interval_data.apply(
                lambda x, intv=interval: x[x <= intv])
            try:
                interval_data = interval_data.apply(
                    lambda x, intv=interval: (intv - x[-1]).total_seconds()
                    if x.size else np.nan)
            except AttributeError:
                interval_data = interval_data.apply(
                    lambda x, intv=interval:
                    (intv - x[-1]) / np.timedelta64(1, 's')
                    if x.size else np.nan)
            x_coord = interval_data * np.cos(activity_df['angle'])
            the_min, the_max = np.min(x_coord), np.max(x_coord)
            x_max = max(abs(the_max), abs(the_min), x_max)
            y_coord = interval_data * np.sin(activity_df['angle'])
            the_min, the_max = np.min(y_coord), np.max(y_coord)
            y_max = max(abs(the_max), abs(the_min), y_max)
            coords = DataFrame({"x": x_coord, "y": y_coord})
            assert interval not in activity_df.columns
            activity_df[interval] = [list(x) for x in coords.values]
        in_secs = {'day': 86400, '2 days': 172800, 'week': 604800,
                   '1 week': 604800, '2 weeks': 1209600, '3 weeks': 1814400,
                   'month': 2635200}
        try:
            xy_max = max(x_max, y_max) / zoom
        except TypeError:
            xy_max = in_secs[zoom]
        except KeyError:
            xy_max = max(x_max, y_max)

        def get_fig(activity_df, col_name):
            """Helper-function returning a fig based on DataFrame and col."""
            plt.style.use(SETTINGS['style'])
            coord = activity_df[col_name].to_dict()
            dists = pd.DataFrame(activity_df[col_name].tolist(),
                                 columns=['x', 'y'],
                                 index=activity_df.index)
            dists = np.sqrt(dists['x']**2 + dists['y']**2)
            in_day = dists[dists < in_secs['day']].count()
            in_week = dists[dists < in_secs['week']].count()
            in_month = dists[dists < in_secs['month']].count()
            fig = plt.figure()
            # left plot (full)
            axes1 = fig.add_subplot(121, aspect='equal')
            axes1.set_xlim([-xy_max, xy_max])
            axes1.set_ylim([-xy_max, xy_max])
            axes1.xaxis.set_ticks([])
            axes1.yaxis.set_ticks([])
            day = plt.Circle((0, 0), in_secs['day'], color='darkslategray')
            week = plt.Circle((0, 0), in_secs['week'], color='slategray')
            month = plt.Circle((0, 0), in_secs['month'], color="lightblue")
            axes1.add_artist(month)
            axes1.add_artist(week)
            axes1.add_artist(day)
            the_date = pd.to_datetime(str(col_name)).strftime(
                '%Y.%m.%d\n%H:%M')
            axes1.text(-xy_max / 1.07, xy_max / 1.30, the_date,
                       bbox=dict(facecolor='slategray', alpha=0.5))
            # axes.text(in_secs['day'], -100, '1 day', fontsize=10)
            # axes.text(in_secs['week'], -100, '1 week', fontsize=10)
            # axes.text(in_secs['month'], -100, '1 month', fontsize=10)
            nx.draw_networkx_nodes(self.graph, coord,
                                   nodelist=activity_df.index.tolist(),
                                   node_color=activity_df['color'],
                                   node_size=20,
                                   cmap=CMAP,
                                   ax=axes1)
            # right plot: zoomed
            axes2 = fig.add_subplot(122, aspect='equal')
            axes2.set_xlim([-xy_max / 10, xy_max / 10])
            axes2.set_ylim([-xy_max / 10, xy_max / 10])
            axes2.xaxis.set_ticks([])
            axes2.yaxis.set_ticks([])
            day = plt.Circle((0, 0), in_secs['day'], color='darkslategray')
            week = plt.Circle((0, 0), in_secs['week'], color='slategray')
            month = plt.Circle((0, 0), in_secs['month'], color="lightblue")
            axes2.add_artist(month)
            axes2.add_artist(week)
            axes2.add_artist(day)
            day_patch = mpatches.Patch(
                color='darkslategray',
                label="{: <3} active in last day".format(in_day).ljust(25))
            week_patch = mpatches.Patch(
                color='slategray',
                label="{: <3} active in last week".format(in_week).ljust(25))
            month_patch = mpatches.Patch(
                color='lightblue',
                label="{: <3} active in last month".format(in_month).ljust(25))
            plt.legend(handles=[day_patch, week_patch, month_patch],
                       loc=1)
            nx.draw_networkx_nodes(self.graph, coord,
                                   nodelist=activity_df.index.tolist(),
                                   node_color=activity_df['color'],
                                   node_size=20,
                                   cmap=CMAP,
                                   ax=axes2)
            return fig

        ion()
        for (num, interval) in enumerate(intervals):
            fig = get_fig(activity_df, interval)
            if show:
                fig.canvas.draw()
                plt.draw()
                plt.pause(1)
                plt.close(fig)
            else:
                plt.savefig("FIGS/img{0:0>5}.png".format(num))
                plt.close(fig)


if __name__ == '__main__':
    PARSER = ac.make_arg_parser(
        ACTIONS.keys(), SETTINGS['project'],
        "Create the author_network of a given project.")
    ARGS = PARSER.parse_args()
    if ARGS.verbose:
        logging.basicConfig(level=getattr(logging, ARGS.verbose.upper()))
    main(ARGS.project,
         do_more=ARGS.more,
         use_cached=ARGS.load,
         cache_it=ARGS.cache,
         delete_all=ARGS.delete)
else:
    logging.basicConfig(level=getattr(logging, SETTINGS['verbose'].upper()))
