"""
Module that includes the AuthorNetwork class,
which has a weighted nx.DiGraph based on a multi_comment_thread,
and a pandas.DataFrame with authors as index.
"""
# Imports
import argparse
from collections import defaultdict
from datetime import datetime
from itertools import combinations
import logging
from math import log
from operator import methodcaller
from textwrap import wrap
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
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
SETTINGS, CMAP = ac.load_settings()

# actions to be used as argument for --more
ACTIONS = {
    "network": "draw_graph",
    "author_activity": "plot_author_activity_bar",
    "author_activity_degree": "plot_activity_degree",
    "author_activity_prop": "plot_activity_prop",
    "centrality_measures": "plot_centrality_measures",
    "histogram": "plot_author_activity_hist",
    "discussion_centre": "draw_centre_discussion"}


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
    def __init__(self, an_mthread):
        super(AuthorNetwork, self).__init__()
        # attributes from argument MultiCommentThread
        self.mthread = an_mthread
        self.all_thread_graphs = an_mthread.graph
        self.node_name = an_mthread.node_name
        self.positions = None  # positions stored when draw_graph is called
        # create author_frame with color as column
        self.author_frame = DataFrame(
            {'color': list(an_mthread.author_color.values())},
            index=list(an_mthread.author_color.keys())).sort_index()
        # add zeros-column for word count
        self.author_frame['word counts'] = np.zeros(
            self.author_frame.shape[0])
        # add zeros-columns for each comment-level (set up to 12)
        for i in range(1, 12):
            self.author_frame["level {}".format(i)] = np.zeros(
                self.author_frame.shape[0])
        # Initialize and populate DiGraph with authors as nodes.
        self.i_graph = nx.DiGraph()
        self.i_graph.add_nodes_from(self.author_frame.index)
        self.c_graph = self.i_graph.to_undirected()
        # generating edges for interaction_graph
        for (source, dest) in self.all_thread_graphs.edges_iter():
            source = self.all_thread_graphs.node[source]['com_author']
            dest = self.all_thread_graphs.node[dest]['com_author']
            if not (source, dest) in self.i_graph.edges():
                self.i_graph.add_weighted_edges_from([(source, dest, 1)])
            else:
                self.i_graph[source][dest]['weight'] += 1
        # Iterate over node-attributes of MultiCommentThread
        # to set values in author_frame and AuthorNetwork
        author_nodes = defaultdict(list)
        author_episodes = defaultdict(set)
        for node, data in self.all_thread_graphs.nodes_iter(data=True):
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
            author_episodes[the_author].add((the_thread, the_cluster))
            # adding timestamp or creating initial list of timestamps for
            # auth in DiGraph
            if 'post_timestamps' in list(self.i_graph.node[the_author].keys()):
                self.i_graph.node[the_author]['post_timestamps'].append(
                    the_date)
            else:
                self.i_graph.node[the_author]['post_timestamps'] = [the_date]
        # create column in author_frame from author_nodes
        self.author_frame["comments"] = Series(
            {key: sorted(value) for (key, value) in author_nodes.items()})
        # create column in author_frame from author_episodes
        self.author_frame["episodes"] = Series(author_episodes)
        # create edges in c_graph
        for source, dest in combinations(author_episodes.keys(), 2):
            overlap = author_episodes[source].intersection(
                author_episodes[dest])
            if overlap:
                self.c_graph.add_edge(source, dest, weight=len(overlap))
        # iterate over node-attributes of AuthorNetwork to sort timestamps
        for _, data in self.i_graph.nodes_iter(data=True):
            data['post_timestamps'] = np.sort(
                np.array(data['post_timestamps'], dtype='datetime64[us]'))
        # removed unused levels-columns in author_frame
        self.author_frame = self.author_frame.loc[
            :, (self.author_frame != 0).any(axis=0)]
        # add columns with total comments and timestamps to author_frame
        self.author_frame['total comments'] = self.author_frame.iloc[
            :, 2:].sum(axis=1)
        self.author_frame['timestamps'] = [self.i_graph.node[an_author][
            "post_timestamps"] for an_author in self.author_frame.index]
        # assert to check that len of comments and timestamps is equal to
        # total comments
        try:
            assert (self.author_frame['comments'].apply(len) ==
                    self.author_frame['total comments']).all()
            assert (self.author_frame['timestamps'].apply(len) ==
                    self.author_frame['total comments']).all()
        except AssertionError as err:
            logging.error("Numbers of comments for %s do not add up: %s",
                          list(self.mthread.thread_url_title.values()), err)
        # adding first and last comment to author_frame
        self.author_frame['first'] = self.author_frame['timestamps'].apply(
            lambda x: x[0])
        self.author_frame['last'] = self.author_frame['timestamps'].apply(
            lambda x: x[-1])
        # generate random angles for each author (to be used in
        # draw_centre_discussion)
        self.author_frame['angle'] = np.linspace(0, 360,
                                                 len(self.author_frame),
                                                 endpoint=False)
        # iterate over authors to create replies columns in author_frame
        for author in self.author_frame.index:
            for label in ['replies (all)',
                          'replies (direct)',
                          'replies (own excl.)']:
                self.author_frame.ix[author, label] = sum(
                    [self.mthread.comment_report(i)[label]
                     for i in self.author_frame.ix[author, "comments"]])
        # adding multiple centrality-measures to author-frame
        self.centrality_measures = {
            'degree centrality': nx.degree_centrality,
            'eigenvector centrality': nx.eigenvector_centrality,
            'page rank': nx.pagerank}
        self.g_types = ["interaction", "cluster"]
        for g_type in self.g_types:
            for measure, function in self.centrality_measures.items():
                graph = self.i_graph if g_type == "interaction" else\
                    self.c_graph
                col = g_type + " " + measure
                try:
                    self.author_frame[col] = Series(function(graph))
                except (ZeroDivisionError, nx.NetworkXError) as err:
                    logging.warning("error with %s: %s", measure, err)
                    self.author_frame[measure] = Series(
                        np.zeros_like(self.author_frame.index))

    def author_count(self):
        """Returns series with count of authors (num of comments per author)"""
        return self.author_frame['total comments']

    def plot_author_activity_bar(self, what='by level', show=True,
                                 project=None,
                                 xfontsize=6):
        """Shows plot of number of comments / wordcount per author.
        what can be either 'by level' or 'word counts' or 'combined'"""
        plt.style.use(SETTINGS['style'])
        if what == "by level":
            cols = self.author_frame.columns[
                self.author_frame.columns.str.startswith('level')]
            levels = self.author_frame[cols].sort_values(
                cols.tolist(), ascending=False)
            total_num_of_comments = int(levels.sum().sum())
            colors = [plt.cm.Set1(20 * i) for i in range(len(levels))]
            axes = levels.plot(kind='barh', stacked=True, color=colors,
                               title='Comment activity (comments) per author (\
                                   total: {})'.format(total_num_of_comments),
                               fontsize=xfontsize)
            axes.set_yticklabels(levels.index, fontsize=xfontsize)
        elif what == "word counts":
            word_counts = self.author_frame[what].sort_values(ascending=False)
            total_word_count = int(word_counts.sum())
            axes = word_counts.plot(
                kind='bar', logy=True,
                title='Comment activity (words) per author (total: {})'.format(
                    total_word_count),
                fontsize=xfontsize)
        elif what == "combined":
            axes = self.author_frame[['total comments', 'word counts']].plot(
                kind='line', logy=True,
                title='Comment activity per author for {}'.format(
                    project).title())
        else:
            raise ValueError
        ac.show_or_save(show)

    def plot_centrality_measures(self, show=True,
                                 project=None,
                                 g_type="cluster",
                                 xfontsize=6):
        """Shows plot of degree_centrality (only for non-zero)"""
        if g_type not in self.g_types:
            raise ValueError
        cols = [g_type + " " + measure for measure in
                self.centrality_measures.keys()]
        centrality = self.author_frame[cols]
        centrality = centrality.sort_values(cols[0],  # TODO: check if best!
                                            ascending=False)
        colors = ["darkslategray", "slategray", "lightblue"]
        plt.style.use(SETTINGS['style'])
        axes = centrality[centrality[cols[0]] != 0].plot(
            kind='bar', color=colors,
            title="Degree centrality, eigenvector-centrality,\
                   and pagerank for {}".format(project).title())
        axes.set_xticklabels(centrality.index, fontsize=xfontsize)
        ac.show_or_save(show)

    def plot_activity_degree(self, show=True, project=None,
                             centrality_measure='eigenvector centrality'):
        """Shows plot of number of comments (bar) and degree-centrality (line)
        for all authors with non-null centrality-measure"""
        if centrality_measure not in set(self.centrality_measures.keys()):
            raise ValueError
        plt.style.use(SETTINGS['style'])
        cols = self.author_frame.columns[
            self.author_frame.columns.str.startswith('level')].tolist()
        measures = [g_type + " " + centrality_measure for
                    g_type in self.g_types]
        data = self.author_frame[cols + measures]
        data = data[data[measures[0]] != 0]
        data = data.sort_values(measures[0], ascending=False)
        colors = [plt.cm.Set1(20 * i) for i in range(len(data.index))]
        axes = data[cols].plot(
            kind='bar', stacked=True, color=colors,
            title="Commenting activity and {} for {}".format(
                centrality_measure, project).title())
        axes.set_ylabel("Number of comments")
        # TODO: add missing legend for marker-types
        axes2 = axes.twinx()
        axes2.set_ylabel(centrality_measure)
        axes2.plot(axes.get_xticks(), data[measures[0]].values,
                   linestyle=':', marker='.', markersize=10, linewidth=.7,
                   color='darkgrey')
        axes2.plot(axes.get_xticks(), data[measures[1]].values,
                   linestyle='-', marker='D', markersize=4, linewidth=.7,
                   color='darkblue')
        ac.show_or_save(show)

    def plot_activity_prop(self, show=True, project=None):
        """Shows plot of number of comments (bar) and proportion
        level-1 / higher-level comment (line) for all authors"""
        plt.style.use(SETTINGS['style'])
        cols = self.author_frame.columns[
            self.author_frame.columns.str.startswith('level')].tolist()
        data = self.author_frame[cols].copy()
        data['proportion'] = (data[cols[1:]].sum(axis=1) /
                              data[cols].sum(axis=1))
        colors = [plt.cm.Set1(20 * i) for i in range(len(data.index))]
        axes = data[cols].plot(
            kind='bar', stacked=True, color=colors,
            title="Commenting activity and proportion\
                of higher-level comments for {}".format(project).title())
        axes.set_ylabel("Number of comments")
        # TODO: add missing legend for marker-types
        axes2 = axes.twinx()
        axes2.set_ylabel("Proportion of Higher-level comments")
        axes2.plot(axes.get_xticks(), data['proportion'].values,
                   linestyle=':', marker='.', markersize=10, linewidth=.7,
                   color='darkgrey')
        ac.show_or_save(show)

    def plot_author_activity_pie(self, what='total comments', show=True,
                                 project=None):
        """Shows plot of commenting activity as piechart
           what can be either 'total comments' (default) or 'word counts'"""
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
        comments = DataFrame({'totals': comments[what].groupby(
            comments.index).sum(),
                              'maxs': comments[what].groupby(
                                  comments.index).max(),
                              'color': comments['color'].groupby(
                                  comments.index).max()}).sort_values(
                                      'maxs', ascending=False)
        for_pie = comments['totals']
        for_pie.name = ""
        norm = mpl.colors.Normalize(vmin=SETTINGS['vmin'],
                                    vmax=SETTINGS['vmax'])
        c_mp = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
        colors = c_mp.to_rgba(comments['color'])
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
            title=('\n'.join(wrap(title, 60))))
        ac.show_or_save(show)

    def plot_author_activity_hist(self, what='total comments', show=True,
                                  project=None):
        """Shows plot of histogram of commenting activity.
           What can be either 'total comments' (default) or 'word counts'"""
        if what not in set(['total comments', 'word counts']):
            raise ValueError
        comments = self.author_frame[what]
        plt.style.use(SETTINGS['style'])
        comments.plot(
            kind='hist',
            bins=50,
            title='Histogram of {} for {}'.format(what, project).title())
        ac.show_or_save(show)

    def w_connected_components(self, graph_type):
        """Returns weakly connected components as generator of list of nodes.
        This ignores the direction of edges."""
        graph = self.c_graph if graph_type == "cluster" else self.i_graph
        return nx.weakly_connected_components(graph)

    def draw_graph(self, graph_type="interaction",
                   project=None, reset=False, show=True):
        """Draws and shows graph."""
        project = None if not project else project
        if graph_type == "cluster":
            graph = self.c_graph
            graph_type = "Co-location Network"
        elif graph_type == "interaction":
            graph = self.i_graph
            graph_type = "Interaction Network"
        # attributing widths and colors to edges
        edges = graph.edges()
        weights = [graph[source][dest]['weight'] * 15 for
                   source, dest in edges]
        edge_colors = [plt.cm.Blues(weight) for weight in weights]
        # attributes sizes to nodes
        sizes = [(log(self.author_count()[author], 4) + 1) * 300
                 for author in self.author_frame.index]
        # positions with spring
        if reset or not self.positions:
            self.positions = nx.spring_layout(graph, k=5, scale=1)
        # creating title and axes
        figure = plt.figure()
        figure.suptitle("{} for {}".format(graph_type, project).title(),
                        fontsize=12)
        axes = figure.add_subplot(111)
        axes.xaxis.set_ticks([])
        axes.yaxis.set_ticks([])
        # actual drawing
        # consider adding legend
        plt.style.use(SETTINGS['style'])
        nx.draw_networkx(graph, self.positions,
                         with_labels=SETTINGS['show_labels_authors'],
                         font_size=7,
                         node_size=sizes,
                         nodelist=self.author_frame.index.tolist(),
                         node_color=self.author_frame['color'].tolist(),
                         edges=edges,
                         width=1,
                         edge_color=edge_colors,
                         vmin=SETTINGS['vmin'],
                         vmax=SETTINGS['vmax'],
                         cmap=CMAP,
                         ax=axes)
        ac.show_or_save(show)

    def draw_centre_discussion(self, regular_intervals=False,
                               project=None,
                               skips=2, zoom=12, show=False):
        """Draws part of nx.DiGraph to picture who's
        at the centre of activity"""
        project = None if not project else project
        df = self.author_frame[['color', 'angle', 'timestamps']].copy()
        if not regular_intervals:
            intervals = np.concatenate(df['timestamps'].values)
            intervals.sort(kind='mergesort')
            intervals = intervals[::skips]
        else:
            start = np.min(df['timestamps'].apply(np.min))
            stop = np.max(df['timestamps'].apply(np.max))
            intervals = date_range(start, stop)[::skips]
        x_max, y_max = 0, 0
        for interval in intervals:
            interval_data = df['timestamps'].apply(
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
            x_coord = interval_data * np.cos(df['angle'])
            the_min, the_max = np.min(x_coord), np.max(x_coord)
            x_max = max(abs(the_max), abs(the_min), x_max)
            y_coord = interval_data * np.sin(df['angle'])
            the_min, the_max = np.min(y_coord), np.max(y_coord)
            y_max = max(abs(the_max), abs(the_min), y_max)
            coords = DataFrame({"x": x_coord, "y": y_coord})
            assert interval not in df.columns
            df[interval] = [list(x) for x in coords.values]
        in_secs = {'day': 86400, '2 days': 172800, 'week': 604800,
                   '1 week': 604800, '2 weeks': 1209600, '3 weeks': 1814400,
                   'month': 2635200}
        try:
            xy_max = max(x_max, y_max) / zoom
        except TypeError:
            xy_max = in_secs[zoom]
        except KeyError:
            xy_max = max(x_max, y_max)

        def get_fig(df, col_name):
            """Helper-function returning a fig based on DataFrame and col."""
            plt.style.use(SETTINGS['style'])
            coord = df[col_name].to_dict()
            dists = pd.DataFrame(df[col_name].tolist(), columns=['x', 'y'],
                                 index=df.index)
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
                                   nodelist=df.index.tolist(),
                                   node_color=df['color'],
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
            plt.legend(handles=[day_patch, week_patch, month_patch])
            nx.draw_networkx_nodes(self.graph, coord,
                                   nodelist=df.index.tolist(),
                                   node_color=df['color'],
                                   node_size=20,
                                   cmap=CMAP,
                                   ax=axes2)
            return fig

        ion()
        for (num, interval) in enumerate(intervals):
            fig = get_fig(df, interval)
            if show:
                fig.canvas.draw()
                plt.draw()
                plt.pause(1)
                plt.close(fig)
            else:
                plt.savefig("FIGS/img{0:0>5}.png".format(num))
                plt.close(fig)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description="Create the author_network of a given project.")
    PARSER.add_argument("project", nargs='?', default=SETTINGS['project'],
                        help="Short name of the project")
    PARSER.add_argument("--more", type=str,
                        choices=ACTIONS.keys(),
                        help="Show output instead of returning object")
    PARSER.add_argument("-l", "--load", action="store_true",
                        help="Load serialized threads when available")
    PARSER.add_argument("-c", "--cache", action="store_true",
                        help="Serialize threads if possible")
    PARSER.add_argument("-v", "--verbose", type=str,
                        choices=['debug', 'info'], default="info",
                        help="Show more logging information")
    PARSER.add_argument("-d", "--delete", action="store_true",
                        help="Delete requests and serialized threads")
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
