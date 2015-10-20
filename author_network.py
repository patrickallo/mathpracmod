"""
Module that includes the author_network class,
which has a weighted nx.DiGraph based on a multi_comment_thread.
"""
# Imports
import joblib
from math import log
from os.path import isfile
import sys
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pandas import DataFrame

import comment_thread as ct
import export_classes as ec

# Loading settings
with open("settings.yaml", "r") as settings_file:
    SETTINGS = yaml.safe_load(settings_file.read())
CMAP = getattr(plt.cm, SETTINGS['cmap'])


THREAD_TYPES = {"Polymath": ct.CommentThreadPolymath,
                "Gilkalai": ct.CommentThreadGilkalai,
                "Gowers": ct.CommentThreadGowers,
                "Terrytao": ct.CommentThreadTerrytao}


# Main
def main(urls, do_more=True):
    """
    Creates AuthorNetwork based on supplied list of urls, and draws graph.
    """
    filename = 'CACHE/' + SETTINGS['filename'] + '_authornetwork.p'
    if isfile(filename):  # authornetwork already saved
        print "loading {}:".format(filename)
        a_network = joblib.load(filename)
        print "complete"
        print "a_network is of type {}".format(type(a_network))
    else:  # authornetwork still to be created
        an_mthread = ct.main(urls, do_more=False)
        filename = 'CACHE/' + SETTINGS['filename'] + '_authornetwork.p'
        a_network = AuthorNetwork(an_mthread)
        print "saving {} as {}:".format(type(a_network), filename),
        joblib.dump(a_network, filename)
        print "complete"
    if do_more:
        print a_network.author_frame.head(n=10)
        a_network.plot_author_count()
        a_network.draw_graph()
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
        graph: weighted nx.DiGraph (weighted edges between authors)

    Methods:
        author_count: returns Counter-object (dict) with
                      authors as keys and num of comments as values
        plot_author_count: plots author_count
        author_report: returns dict with comments,
                       replies and direct replies,
                       and comments per level for a given author
        weakly connected components: returns generator of
                                     weakly connected components
        draw_graph: draws author_network

    """

    def __init__(self, an_mthread):
        super(AuthorNetwork, self).__init__()
        self.mthread = an_mthread
        self.all_thread_graphs = an_mthread.graph
        self.node_name = an_mthread.node_name
        self.author_frame = DataFrame({
            'color': an_mthread.author_color.values()},
            index=an_mthread.author_color.keys()).sort_index()
        for i in range(1, 6):
            self.author_frame["level {}".format(i)] = np.zeros(
                len(self.author_frame.index))
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
            self.author_frame.ix[the_author, the_level] += 1
            if 'post_timestamps' in self.graph.node[the_author].keys():
                self.graph.node[the_author]['post_timestamps'].append(the_date)
            else:
                self.graph.node[the_author]['post_timestamps'] = [the_date]
        for _, data in self.graph.nodes_iter(data=True):
            data['post_timestamps'].sort()
        self.author_frame = self.author_frame.loc[
            :, (self.author_frame != 0).any(axis=0)]
        self.author_frame['total comments'] = self.author_frame.iloc[
            :, 1:].sum(axis=1)
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

    def plot_author_count(self, show=True):
        """Shows plot of number of comments per author"""
        cols = self.author_frame.columns[
            self.author_frame.columns.str.startswith('level')]
        levels = self.author_frame[cols]
        plt.style.use('ggplot')
        levels.plot(kind='bar', stacked=True,
                    title='Comment activity per author')
        if show:
            plt.show()
        else:
            filename = raw_input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def w_connected_components(self):
        """Returns weakly connected components as generator of list of nodes.
        This ignores the direction of edges."""
        return nx.weakly_connected_components(self.graph)

    def draw_graph(
            self,
            title="Author network for " + SETTINGS['filename'],
            show=True):
        """Draws and shows graph."""
        # attributing widths to edges
        edges = self.graph.edges()
        weights = [self.graph[source][dest]['weight'] / float(10) for
                   source, dest in edges]
        # attributes sizes to nodes
        # TODO: try to make size proportional to total len of posts
        sizes = [(log(self.author_count()[author], 4) + 1) * 300
                 for author in self.author_frame.index]
        # positions with spring
        positions = nx.spring_layout(self.graph, k=2.5, scale=1)
        # creating title and axes
        figure = plt.figure()
        figure.suptitle(title, fontsize=12)
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
        plt.style.use('ggplot')
        if show:
            plt.show()
        else:
            filename = raw_input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)


if __name__ == '__main__':
    ARGUMENTS = sys.argv[1:]
    if ARGUMENTS:
        main(ARGUMENTS)
    else:
        print SETTINGS['msg']
        main(SETTINGS['urls'])
