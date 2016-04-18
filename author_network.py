"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a multi_comment_thread.
"""
# Imports
from math import log
from textwrap import wrap
import sys
import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame, date_range, Series
from pandas.tools.plotting import parallel_coordinates
from pylab import ion

import comment_thread as ct
import export_classes as ec

# Loading settings
with open("settings.yaml", "r") as settings_file:
    SETTINGS = yaml.safe_load(settings_file.read())
CMAP = getattr(plt.cm, SETTINGS['cmap'])


THREAD_TYPES = {"Polymathprojects": ct.CommentThreadPolymath,
                "Gilkalai": ct.CommentThreadGilkalai,
                "Gowers": ct.CommentThreadGowers,
                "Sbseminar": ct.CommentThreadSBSeminar,
                "Terrytao": ct.CommentThreadTerrytao}


# Main
def main(project, do_more=True, use_cached=False, cache_it=False):
    """
    Creates AuthorNetwork based on supplied list of urls, and draws graph.
    """
    try:
        an_mthread = ct.main(project, do_more=False,
                             use_cached=use_cached, cache_it=cache_it)
    except:
        print("Could not create mthread")
        sys.exit(1)
    a_network = AuthorNetwork(an_mthread)
    if do_more:
        # a_network.plot_author_activity_bar(what="by level")
        # a_network.plot_degree_centrality()
        # a_network.plot_centrality_counts()
        # a_network.plot_activity_degree()
        # a_network.plot_author_activity_bar()
        # a_network.plot_author_activity_pie(what="total comments")
        # a_network.plot_author_activity_pie(what="word counts")
        # a_network.plot_author_activity_hist()
        # a_network.plot_author_activity_hist(what='word counts')
        a_network.draw_graph()
        # print(a_network.author_frame)
        # a_network.draw_centre_discussion(reg_intervals=False,
        #                                skips=10, zoom='2 weeks',
        #                                show=False)
    else:
        return a_network


# Classes
class AuthorNetwork(ec.GraphExportMixin, object):

    """
    Creates and draws Weighted nx.DiGraph of comments between authors.

    Attributes:
        mthread: ct.MultiCommentThread object.
        all_thread_graphs: DiGraph of ct.MultiCommentThread.
        node_name: dict with nodes as keys and authors as values.
        author_frame: pandas.DataFrame with authors as index.
        graph: weighted nx.DiGraph (weighted edges between authors)

    Methods:
        author_count: returns Counter-object (dict) with
                      authors as keys and num of comments as values
        plot_author_activity_bar: plots commenting activity in bar-chart
        plot_author_activity_pie: plots commenting activity in pie-chart
        plot_author_activity_hist: plots histogram of commenting activity
        weakly connected components: returns generator of
                                     weakly connected components
        draw_graph: draws author_network

    """
    def __init__(self, an_mthread):
        super(AuthorNetwork, self).__init__()
        self.mthread = an_mthread
        self.all_thread_graphs = an_mthread.graph
        self.node_name = an_mthread.node_name
        self.author_frame = DataFrame(
            {'color': list(an_mthread.author_color.values())},
            index=list(an_mthread.author_color.keys())).sort_index()
        self.author_frame['word counts'] = np.zeros(
            self.author_frame.shape[0])
        for i in range(1, 12):
            self.author_frame["level {}".format(i)] = np.zeros(
                self.author_frame.shape[0])
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.author_frame.index)
        for (source, dest) in self.all_thread_graphs.edges_iter():
            source = self.all_thread_graphs.node[source]['com_author']
            dest = self.all_thread_graphs.node[dest]['com_author']
            if not (source, dest) in self.graph.edges():
                self.graph.add_weighted_edges_from([(source, dest, 1)])
            else:
                self.graph[source][dest]['weight'] += 1
        for _, data in self.all_thread_graphs.nodes_iter(data=True):
            # set comment_levels in author_frame, and
            # set data for first and last comment in self.graph
            the_author = data['com_author']
            the_level = 'level {}'.format(data['com_depth'])
            the_date = data['com_timestamp']
            the_count = len(data['com_tokens'])
            self.author_frame.ix[the_author, the_level] += 1
            self.author_frame.ix[the_author, 'word counts'] += the_count
            # adding timestamp or creating initial list of timestamps for auth
            if 'post_timestamps' in list(self.graph.node[the_author].keys()):
                self.graph.node[the_author]['post_timestamps'].append(the_date)
            else:
                self.graph.node[the_author]['post_timestamps'] = [the_date]
        for _, data in self.graph.nodes_iter(data=True):
            data['post_timestamps'] = np.sort(
                np.array(data['post_timestamps'], dtype='datetime64[us]'))
        self.author_frame = self.author_frame.loc[
            :, (self.author_frame != 0).any(axis=0)]
        self.author_frame['total comments'] = self.author_frame.iloc[
            :, 2:].sum(axis=1)
        self.author_frame['timestamps'] = [self.graph.node[an_author][
            "post_timestamps"] for an_author in self.author_frame.index]
        self.author_frame['first'] = self.author_frame['timestamps'].apply(
            lambda x: x[0])
        self.author_frame['last'] = self.author_frame['timestamps'].apply(
            lambda x: x[-1])
        self.author_frame['angle'] = np.linspace(0, 360,
                                                 len(self.author_frame),
                                                 endpoint=False)
        for author in self.author_frame.index:
            the_comments = [node_id for (node_id, data) in
                            self.all_thread_graphs.nodes_iter(data=True) if
                            data["com_author"] == author]
            for label in ['replies (all)',
                          'replies (direct)',
                          'replies (own excl.)']:
                self.author_frame.ix[author, label] = sum(
                    [self.mthread.comment_report(i)[label]
                     for i in the_comments])
        try:
            self.author_frame['degree centrality'] = Series(
                nx.degree_centrality(self.graph))
        except ZeroDivisionError as err:
            print("error with degree centrality: ", err)
            self.author_frame['degree centrality'] = Series(
                np.zeros_like(self.author_frame.index))
        try:
            self.author_frame['eigenvector centrality'] = Series(
                nx.eigenvector_centrality(self.graph))
        except (ZeroDivisionError, nx.NetworkXError) as err:
            print("error with eigenvector centrality:", err)
            self.author_frame['eigenvector centrality'] = Series(
                np.zeros_like(self.author_frame.index))
        try:
            self.author_frame['page rank'] = Series(
                nx.pagerank(self.graph))
        except ZeroDivisionError as err:
            print("error with page rank: ", err)
            self.author_frame['page rank'] = Series(
                np.zeros_like(self.author_frame.index))

    def author_count(self):
        """Returns series with count of authors (num of comments per author)"""
        return self.author_frame['total comments']

    def plot_author_activity_bar(self, what='by level', show=True,
                                 project=SETTINGS['msg'],
                                 xfontsize=6):
        """Shows plot of number of comments / wordcount per author.
        what can be either 'by level' or 'word counts' or 'combined'"""
        if what not in set(['by level', 'word counts', 'combined']):
            raise ValueError
        plt.style.use(SETTINGS['style'])
        if what == "by level":
            cols = self.author_frame.columns[
                self.author_frame.columns.str.startswith('level')]
            levels = self.author_frame[cols].sort_values(
                cols.tolist(), ascending=False)
            axes = levels.plot(kind='barh', stacked=True,
                               title='Comment activity (comments) per author')
            axes.set_yticklabels(levels.index, fontsize=xfontsize)
        elif what == "word counts":
            axes = self.author_frame[what].plot(
                kind='bar', logy=True,
                title='Comment activity (words) per author')
        elif what == "combined":
            axes = self.author_frame[['total comments', 'word counts']].plot(
                kind='line', logy=True,
                title='Comment activity per author for {}'.format(
                    project).title())
        else:
            pass
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_centrality_measures(self, show=True,
                                 project=SETTINGS['msg'],
                                 xfontsize=6):
        """Shows plot of degree_centrality (only for non-zero)"""
        centrality = self.author_frame[['degree centrality',
                                        'eigenvector centrality',
                                        'page rank']]
        centrality = centrality.sort_values('degree centrality',
                                            ascending=False)
        plt.style.use(SETTINGS['style'])
        axes = centrality[centrality['degree centrality'] != 0].plot(
            kind='bar',
            title="Degree centrality, eigenvector-centrality,\
                   and pagerank for {}".format(project).title())
        axes.set_xticklabels(centrality.index, fontsize=xfontsize)
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_centrality_counts(self, show=True, project=SETTINGS['msg']):
        """Plots different centrality-measures with parallel-coordinates"""
        data = self.author_frame[['total comments',
                                  'degree centrality',
                                  'eigenvector centrality',
                                  'page rank']]
        comments_range = np.arange(
            0, data['total comments'].max()+50, 50)
        data.loc[:, 'ranges'] = pd.cut(data['total comments'], comments_range)
        del data['total comments']
        plt.figure()
        plt.suptitle("Comparison of centrality-measures for {}".format(
            project).title())
        parallel_coordinates(data, 'ranges')
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_activity_degree(self, show=True, project=SETTINGS['msg']):
        """Shows plot of number of comments (bar) and degree-centrality (line)
        for all authors"""
        plt.style.use(SETTINGS['style'])
        cols = self.author_frame.columns[
            self.author_frame.columns.str.startswith('level')].tolist()
        data = self.author_frame[cols + ['degree centrality']]
        data = data[data['degree centrality'] != 0]
        colors = [plt.cm.Set1(20*i) for i in range(len(data.index))]
        axes = data[cols].plot(
            kind='bar', stacked=True, color=colors,
            title="activity and degree centrality for {}".format(
                project).title())
        axes.set_ylabel("Number of comments")
        axes2 = axes.twinx()
        axes2.set_ylabel("Degree centrality")
        axes2.plot(axes.get_xticks(), data['degree centrality'].values,
                   linestyle=':', marker='.', linewidth=.5,
                   color='grey')
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_author_activity_pie(self, what='total comments', show=True,
                                 project=SETTINGS['msg']):
        """Shows plot of commenting activity as piechart
           what can be either 'total comments' (default) or 'word counts'"""
        if what not in set(['total comments', 'word counts']):
            raise ValueError
        comments = self.author_frame[[what, 'color']].sort_values(
            what, ascending=False)
        thresh = int(np.ceil(comments[what].sum()/100))
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
                                  comments.index).max()}
                            ).sort_values('maxs', ascending=False)
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
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_author_activity_hist(self, what='total comments', show=True,
                                  project=SETTINGS['msg']):
        """Shows plit of histogram of commenting activity.
           What can be either 'total comments' (default) or 'word counts'"""
        if what not in set(['total comments', 'word counts']):
            raise ValueError
        comments = self.author_frame[what]
        plt.style.use(SETTINGS['style'])
        comments.plot(
            kind='hist',
            bins=50,
            title='Histogram of {} for {}'.format(what, project).title())
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def w_connected_components(self):
        """Returns weakly connected components as generator of list of nodes.
        This ignores the direction of edges."""
        return nx.weakly_connected_components(self.graph)

    def draw_graph(self, project=SETTINGS['msg'], show=True):
        """Draws and shows graph."""
        # attributing widths to edges
        edges = self.graph.edges()
        weights = [self.graph[source][dest]['weight'] / float(10) for
                   source, dest in edges]
        # attributes sizes to nodes
        sizes = [(log(self.author_count()[author], 4) + 1) * 300
                 for author in self.author_frame.index]
        # positions with spring
        positions = nx.spring_layout(self.graph, k=None, scale=1)
        # creating title and axes
        figure = plt.figure()
        figure.suptitle("Author network for {}".format(project).title(),
                        fontsize=12)
        axes = figure.add_subplot(111)
        axes.xaxis.set_ticks([])
        axes.yaxis.set_ticks([])
        # actual drawing
        plt.style.use(SETTINGS['style'])
        nx.draw_networkx(self.graph, positions,
                         with_labels=SETTINGS['show_labels_authors'],
                         font_size=7,
                         node_size=sizes,
                         nodelist=self.author_frame.index.tolist(),
                         node_color=self.author_frame['color'].tolist(),
                         edges=edges,
                         width=weights,
                         vmin=SETTINGS['vmin'],
                         vmax=SETTINGS['vmax'],
                         cmap=CMAP,
                         ax=axes)
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def draw_centre_discussion(self, reg_intervals=False,
                               skips=1, zoom=1, show=True):
        """Draws part of nx.DiGraph to picture who's
        at the centre of activity"""
        df = self.author_frame[['color', 'angle', 'timestamps']]
        if not reg_intervals:
            intervals = np.concatenate(df['timestamps'].values)
            intervals.sort(kind='mergesort')
            intervals = intervals[::skips]
        else:
            start = np.min(df['timestamps'].apply(np.min))
            stop = np.max(df['timestamps'].apply(np.max))
            intervals = date_range(start, stop)[::skips]
        x_max, y_max = 0, 0
        for interval in intervals:
            df[interval] = df['timestamps'].apply(
                lambda x, intv=interval: x[x <= intv])
            try:
                df[interval] = df[interval].apply(
                    lambda x, intv=interval: (intv - x[-1]).total_seconds()
                    if x.size else np.nan)
            except AttributeError:
                df[interval] = df[interval].apply(
                    lambda x, intv=interval:
                    (intv - x[-1]) / np.timedelta64(1, 's')
                    if x.size else np.nan)
            x_coord = df[interval] * np.cos(df['angle'])
            the_min, the_max = np.min(x_coord), np.max(x_coord)
            x_max = max(abs(the_max), abs(the_min), x_max)
            y_coord = df[interval] * np.sin(df['angle'])
            the_min, the_max = np.min(y_coord), np.max(y_coord)
            y_max = max(abs(the_max), abs(the_min), y_max)
            coords = DataFrame({"x": x_coord, "y": y_coord})
            df[interval] = [tuple(x) for x in coords.values]
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
            df = df[df[col_name] != (np.nan, np.nan)]
            coord = df[col_name].to_dict()
            dists = df[col_name].apply(lambda x: np.sqrt(x[0]**2 + x[1]**2))
            in_day = dists[dists < in_secs['day']].count()
            in_week = dists[dists < in_secs['week']].count()
            in_month = dists[dists < in_secs['month']].count()
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            axes.set_xlim([-xy_max, xy_max])
            axes.set_ylim([-xy_max, xy_max])
            axes.xaxis.set_ticks([])
            axes.yaxis.set_ticks([])
            day = plt.Circle((0, 0), in_secs['day'], color='darkslategray')
            week = plt.Circle((0, 0), in_secs['week'], color='slategray')
            month = plt.Circle((0, 0), in_secs['month'], color="lightblue")
            axes.add_artist(month)
            axes.add_artist(week)
            axes.add_artist(day)
            the_date = pd.to_datetime(str(col_name)).strftime(
                '%Y.%m.%d\n%H:%M')
            axes.text(-xy_max/1.07, xy_max/1.26, the_date,
                      bbox=dict(facecolor='slategray', alpha=0.5))
            # axes.text(in_secs['day'], -100, '1 day', fontsize=10)
            # axes.text(in_secs['week'], -100, '1 week', fontsize=10)
            # axes.text(in_secs['month'], -100, '1 month', fontsize=10)
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
                                   ax=axes)
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
    ARGUMENTS = sys.argv[1:]
    if ARGUMENTS:
        SETTINGS['filename'] = input("Filename to be used: ")
        SETTINGS['msg'] = input("Message to be used: ")
        main(ARGUMENTS)
    else:
        print(SETTINGS['msg'])
        main(SETTINGS['project'], use_cached=False, cache_it=False)
