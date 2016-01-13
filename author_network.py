"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a multi_comment_thread.
"""
# Imports
import joblib
from math import log
from os.path import isfile
from os import remove
from glob import iglob
import sys
import yaml
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
from pandas import DataFrame, Series

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
def main(urls, do_more=True, use_cached=False, cache_it=False):
    """
    Creates AuthorNetwork based on supplied list of urls, and draws graph.
    """
    filename = 'CACHE/' + SETTINGS['filename'] + '_authornetwork.p'
    if use_cached and isfile(filename):  # authornetwork already saved
        print("loading {}:".format(filename))
        a_network = joblib.load(filename)
        print("complete")
        print("a_network is of type {}".format(type(a_network)))
    else:  # authornetwork still to be created
        if isfile(filename):
            for to_delete in iglob(filename + '*'):
                remove(to_delete)
        an_mthread = ct.main(urls, do_more=False,
                             use_cached=use_cached, cache_it=cache_it)
        filename = 'CACHE/' + SETTINGS['filename'] + '_authornetwork.p'
        a_network = AuthorNetwork(an_mthread)
        if cache_it:
            print("saving {} as {}:".format(type(a_network),
                                            filename), end=' ')
            joblib.dump(a_network, filename)
            print("complete")
    if do_more:
        # a_network.plot_author_activity_bar(what="by level")
        # a_network.plot_author_activity_bar(what="word counts")
        # a_network.plot_author_activity_pie(what="total comments")
        # a_network.plot_author_activity_pie(what="word counts")
        # a_network.plot_author_activity_hist()
        # a_network.plot_author_activity_hist(what='word counts')
        # a_network.draw_graph()
        print(a_network.author_frame['level 4'].cumsum())
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
            if 'post_timestamps' in list(self.graph.node[the_author].keys()):
                self.graph.node[the_author]['post_timestamps'].append(the_date)
            else:
                self.graph.node[the_author]['post_timestamps'] = [the_date]
        for _, data in self.graph.nodes_iter(data=True):
            data['post_timestamps'].sort()
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

    def author_count(self):
        """Returns series with count of authors (num of comments per author)"""
        return self.author_frame['total comments']

    def plot_author_activity_bar(self, what='by level', show=True,
                                 project=SETTINGS['msg']):
        """Shows plot of number of comments / wordcount per author.
        what can be either 'by level' or 'word counts' or 'combined'"""
        if what not in set(['by level', 'word counts', 'combined']):
            raise ValueError
        plt.style.use(SETTINGS['style'])
        if what == "by level":
            cols = self.author_frame.columns[
                self.author_frame.columns.str.startswith('level')]
            levels = self.author_frame[cols]
            levels.plot(kind='bar', stacked=True,
                        title='Comment activity (comments) per author')
        elif what == "word counts":
            self.author_frame[what].plot(
                kind='bar', logy=True,
                title='Comment activity (words) per author')
        elif what == "combined":
            self.author_frame[['total comments', 'word counts']].plot(
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

    def plot_degree_centrality(self, show=True, project=SETTINGS['msg']):
        """Shows hist of degree_centrality (only for non-zero)"""
        deg_centrality = Series(nx.degree_centrality(self.graph))
        plt.style.use(SETTINGS['style'])
        deg_centrality[deg_centrality != 0].plot(
            kind='bar',
            title="Degree centrality for {}".format(project).title())
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
        positions = nx.spring_layout(self.graph, k=2.5, scale=1)
        # creating title and axes
        figure = plt.figure()
        figure.suptitle("Author network for {}".format(project).title(),
                        fontsize=12)
        axes = figure.add_subplot(111)
        axes.xaxis.set_ticks([])
        axes.yaxis.set_ticks([])
        # actual drawing
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
        plt.style.use(SETTINGS['style'])
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)


if __name__ == '__main__':
    ARGUMENTS = sys.argv[1:]
    if ARGUMENTS:
        SETTINGS['filename'] = input("Filename to be used: ")
        SETTINGS['msg'] = input("Message to be used: ")
        main(ARGUMENTS)
    else:
        print(SETTINGS['msg'])
        main(SETTINGS['urls'], use_cached=True, cache_it=True)
