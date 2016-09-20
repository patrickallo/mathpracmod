
"""
Module that includes the MultiCommentThread class
"""

# Imports
from collections import defaultdict, OrderedDict
import datetime
import logging

from matplotlib.dates import date2num, DateFormatter, DayLocator, MonthLocator
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from pandas import DataFrame


import access_classes as ac
import export_classes as ec

# Loading settings
SETTINGS, CMAP = ac.load_settings()


class MultiCommentThread(ac.ThreadAccessMixin, ec.GraphExportMixin, object):
    """
    Combines graphs of multiple comment_threads for uniform author colouring.
    Main uses: drawing graphs, and supply to author_network.
    Drawing of separate threads should also use this class.
    Inherits methods from two parent Mixins.

    Attributes:
        graph: nx.DiGraph (overruled from ThreadAccessMixin).
        author_color: dict with authors as keys and colors (ints) as values.
        node_name: dict with nodes as keys and authors as values
        type_nodes: defaultdict with thread_class as key and
                   list of nodes (comments) as values
        thread_url_title: OrderedDict with url:post_titles of
                          the respective threads
        corpus: list of unicode-strings (one per comment)
        vocab: dict with:
            tokenized: flat list of tokens (of all comments)
            stemmed: flat list of stems (of all comments)
            frame: pandas.DataFrame with vocab_tokenized as 'word' columns,
                     and vocab_stemmed as index

    Methods:
        add_thread: mutator method for adding thread to multithread.
                    This method is called by init.
        draw_graph: accessor method that draws the mthread graph.
        __plot_activity: helper-method called by plot_activity_...
        plot_activity_author: accessor method plotting of
            x-axis: time_stamps
            y-axis: authors
        plot_activity_thread: accessor method plotting of
            x-axis: time_stamps
            y-axis: threads
        plot_growth: accessor method plotting of
            x-axis:  time_stamps
            y-axis: cummulative word-count
        from ThreadAccessMixin:
            comment_report: takes node-id(s), and
                            returns dict with report about node.
            print_nodes: takes nodes-id(s), and
                         prints out node-data as yaml. No output.
            (two more unused methods)
        from ThreadExportMixin:
            to_gephi: exports the full graph to gephi.
            to_yaml: exports the full graph to yaml.

    """

    def __init__(self, *threads):
        super(MultiCommentThread, self).__init__()
        self.author_color = {}
        self.node_name = {}
        self.type_nodes = defaultdict(list)
        self.thread_url_title = OrderedDict()
        self.corpus = []
        self.vocab = {'tokenized': [],
                      'stemmed': []}
        for thread in threads:
            self.add_thread(thread, replace_frame=False)
            self.type_nodes[thread.__class__.__name__] += thread.graph.nodes()
        self.vocab['frame'] = DataFrame({'words': self.vocab['tokenized']},
                                        index=self.vocab['stemmed'])

    # Mutator methods
    def add_thread(self, thread, replace_frame=True):
        """
        Adds new (non-overlapping) thread by updating author_color and DiGraph.
        """
        # step 1: updating of lists and dicts
        new_authors = thread.authors.difference(self.author_color.keys())
        new_colors = {a: c for (a, c) in
                      zip(new_authors,
                          range(len(self.author_color),
                                len(self.author_color) + len(new_authors)))}
        self.author_color.update(new_colors)
        # assert tests for non-overlap of node_id's between threads
        try:
            overlap = set(self.node_name.keys()).intersection(
                set(thread.node_name.keys()))
            assert not overlap
        except AssertionError:
            logging.warning(
                "Overlapping threads found when adding %s.\n\
                Overlapping nodes: %s", thread.post_title, overlap)
        self.node_name.update(thread.node_name)
        self.thread_url_title[thread.data.thread_url] = thread.post_title
        # step 2: updating vocabularies
        for _, data in thread.graph.nodes_iter(data=True):
            self.corpus.append(data["com_content"])
            self.vocab['tokenized'].extend(data["com_tokens"])
            self.vocab['stemmed'].extend(data["com_stems"])
        if replace_frame:  # only when called outside init
            self.vocab['frame'] = DataFrame({'words': self.vocab['tokenized']},
                                            index=self.vocab['stemmed'])
        # step 3: composing graphs
        self.graph = nx.compose(self.graph, thread.graph)

    # Helper methods
    def __count_activity(self):
        stamps, thread, thread_type, author, wordcounts = zip(
            *((data["com_timestamp"],
               data["com_thread"],
               data["com_thread"].netloc.split('.')[0],
               data["com_author"],
               len(data["com_tokens"]))
              for _, data in self.graph.nodes_iter(data=True)))
        growth = DataFrame(
            {'wordcounts': wordcounts,
             'thread': thread,
             'thread type': thread_type,
             'author': author},
            index=stamps)
        growth = growth.sort_index()
        return growth

    @staticmethod
    def __plot_activity(items, tick_tuple, start, stop, first, last,
                        **kwargs):
        """
        Plots and shows (alt: saves) plot of
            x-axis: time_stamps,
            y-axis: what's active (author / thread).
        Set project as kwarg for correct title
        """
        # Setup the plot
        plt.title("{} activity over time for {}".format(
            kwargs.get("activity", None),
            kwargs.get("project", None)).title(), fontsize=12)
        plt.style.use(SETTINGS['style'])
        axes = plt.gca()
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_major_locator(MonthLocator(
            interval=kwargs.get('intervals')))
        fontsize = 4 if len(items) >= 15 else 6
        axes.set_yticklabels(items, fontsize=fontsize)
        first, last, *_ = ac.check_date_type(first, last)
        plt.xlim(max([start, first]),
                 min([stop, last]))
        plt.yticks(range(1, len(items) + 1), tick_tuple)
        ac.show_or_save(kwargs.get("show", True))

    def __color_by_cluster(self, items, key, start, stop):
        norm = mpl.colors.Normalize(vmin=SETTINGS['vmin'],
                                    vmax=SETTINGS['vmax'])
        c_mp = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Set1)
        for y_value, item in enumerate(items, start=1):
            timestamp_cluster = [
                (data["com_timestamp"], data["cluster_id"])
                for (_, data) in self.graph.nodes_iter(data=True)
                if data[key] == item]
            if timestamp_cluster:
                timestamps, _ = list(zip(*timestamp_cluster))
                this_start, this_stop = min(timestamps), max(timestamps)
                start, stop = min(start, this_start), max(stop, this_stop)
                plt.hlines(y_value, this_start, this_stop, 'k', lw=.5)
                for timestamp, cluster in timestamp_cluster:
                    v_color = c_mp.to_rgba(cluster * 15)
                    plt.vlines(timestamp,
                               y_value + 0.05, y_value - 0.05,
                               v_color, lw=1)
            else:
                logging.warning("Plotting failed due to empty threads")
                return
        return start, stop

    # Accessor methods
    def draw_graph(self,
                   intervals=10, first=None, last=None,
                   **kwargs):
        """Draws and shows (alt: saves) DiGraph of MultiCommentThread
        as tree-structure.
        Should be called with project as kwarg for correct title."""
        project, show, _ = ac.handle_kwargs(**kwargs)
        first = SETTINGS['first_date'] if not first else first
        last = SETTINGS['last_date'] if not last else last
        # creating title and axes
        figure = plt.figure()
        figure.suptitle("Thread structure for {}".format(project).title(),
                        fontsize=12)
        axes = figure.add_subplot(111)
        axes.yaxis.set_major_locator(DayLocator(interval=intervals))
        axes.yaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_ticks(list(range(1, 7)))
        axes.set_xlabel("Comment Levels")
        first, last, *_ = ac.check_date_type(first, last)
        dates = sorted([data["com_timestamp"] for _, data in
                        self.graph.nodes_iter(data=True)])
        first, last = max(first, dates[0]), min(last, dates[-1])
        plt.ylim(first, last)
        # creating and drawingsub_graphs
        types_markers = {thread_type: marker for (thread_type, marker) in
                         zip(self.type_nodes.keys(),
                             ['o', '>', 'H', 'D'][:len(
                                 self.type_nodes.keys())])}
        for (thread_type, marker) in types_markers.items():
            type_subgraph = self.graph.subgraph(self.type_nodes[thread_type])
            # generating colours and positions for sub_graph
            positions = {node_id: (data["com_depth"],
                                   date2num(data["com_timestamp"]))
                         for (node_id, data) in
                         type_subgraph.nodes_iter(data=True)}
            node_color = {node_id: (self.author_color[self.node_name[node_id]])
                          for node_id in type_subgraph.nodes()}
            # drawing nodes of type_subgraph
            nx.draw_networkx_nodes(type_subgraph, positions,
                                   node_size=20,
                                   nodelist=list(node_color.keys()),
                                   node_color=list(node_color.values()),
                                   node_shape=marker,
                                   vmin=SETTINGS['vmin'],
                                   vmax=SETTINGS['vmax'],
                                   cmap=CMAP,
                                   ax=axes)
            nx.draw_networkx_edges(type_subgraph, positions, width=.5)
            if SETTINGS['show_labels_comments']:
                nx.draw_networkx_labels(
                    type_subgraph, positions, font_size=8,
                    labels={node: node[9:] for node in node_color.keys()})
        # show all
        plt.style.use(SETTINGS['style'])
        the_lines = [mlines.Line2D([], [], color='gray',
                                   marker=marker,
                                   markersize=5,
                                   label=thread_type[13:])
                     for (thread_type, marker) in types_markers.items()]
        plt.legend(title="Where is the discussion happening",
                   handles=the_lines)
        ac.show_or_save(show)

    def plot_activity_thread(self,
                             color_by="cluster",
                             intervals=1, first=None, last=None,
                             **kwargs):
        """
        Plots and shows (alt: saves) plot of
            x-axis: time_stamps,
            y-axis: thread.
        Colours can be based on clusters and on authors.
        Set project as kwarg for correct title
        """
        project, show, _ = ac.handle_kwargs(**kwargs)
        first = SETTINGS['first_date'] if not first else first
        last = SETTINGS['last_date'] if not last else last
        stop = datetime.datetime(2000, 1, 1)
        start = datetime.datetime.now()
        items = list(self.thread_url_title.keys())
        tick_tuple = tuple([item.netloc + "\n" + item.path for
                            item in items])
        key = "com_thread"
        if color_by.lower() == "cluster":
            start, stop = self.__color_by_cluster(items, key, start, stop)
        elif color_by.lower() == "author":
            for y_value, item in enumerate(items, start=1):
                timestamp_author = [
                    (data["com_timestamp"], data["com_author"])
                    for (_, data) in self.graph.nodes_iter(data=True)
                    if data[key] == item]
                timestamps, _ = list(zip(*timestamp_author))
                this_start, this_stop = min(timestamps), max(timestamps)
                start, stop = min(start, this_start), max(stop, this_stop)
                plt.hlines(y_value, this_start, this_stop, 'k', lw=.5)
                for timestamp, author in timestamp_author:
                    v_color = ac.color_list(self.author_color[author],
                                            SETTINGS['vmin'], SETTINGS['vmax'],
                                            cmap=CMAP,)
                    plt.vlines(timestamp,
                               y_value + 0.05, y_value - 0.05,
                               v_color, lw=1)
        else:
            raise ValueError
        self.__plot_activity(items, tick_tuple, start, stop, first, last,
                             activity='thread', intervals=intervals,
                             show=show, project=project)

    def plot_activity_author(self,
                             intervals=1, first=None, last=None,
                             **kwargs):
        """
        Plots and shows (alt: saves) plot of
            x-axis: time_stamps,
            y-axis: author.
        Colours are always by cluster.
        Set project as kwarg for correct title
        """
        project, show, _ = ac.handle_kwargs(**kwargs)
        stop = datetime.datetime(2000, 1, 1)
        start = datetime.datetime.now()
        items = list(self.author_color.keys())
        tick_tuple = tuple(items)
        key = "com_author"
        start, stop = self.__color_by_cluster(items, key, start, stop)
        self.__plot_activity(items, tick_tuple, start, stop, first, last,
                             activity='author', intervals=intervals,
                             show=show, project=project)

    def plot_growth_size(self, show_counts=False, **kwargs):
        """Plots and shows (alt: saves) barplot of
        number of comments per week"""
        project, show, _ = ac.handle_kwargs(**kwargs)
        data = self.__count_activity()['wordcounts'].resample('W')
        data = data.agg(['sum', 'count'])
        axes = plt.figure().add_subplot(111)
        plt.style.use(SETTINGS['style'])
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(DateFormatter("%b %d\n%Y"))
        axes.set_ylabel("Wordcounts")
        axes.set_title("Weekly commenting in {}".format(project))
        axes.bar(data.index, data['sum'], label="wordcounts")
        if show_counts:
            axes2 = axes.twinx()
            axes2.set_ylabel("Number of comments")
            for tlabel in axes2.get_yticklabels():
                tlabel.set_color('firebrick')
            axes2.plot(data.index, data['count'], label="number of comments")
        ac.show_or_save(show)

    def plot_growth(self,
                    plot_by='thread type',
                    first=SETTINGS['first_date'], last=SETTINGS['last_date'],
                    **kwargs):
        """Plots and shows (alt: saves) how fast a thread grows
        (cumsum of wordcounts)
        Set project as kwarg for correct title"""
        project, show, fontsize = ac.handle_kwargs(**kwargs)
        first = SETTINGS['first_date'] if not first else first
        last = SETTINGS['last_date'] if not last else last
        try:
            growth = self.__count_activity()[['wordcounts', plot_by]]
        except KeyError:
            raise ValueError("By is either thread type, thread or author")
        # grouping on index to clean duplicate timestamps
        # asssumption: same timestamp is also same blog + same timestamp does
        # not happen with author
        growth = growth.groupby(growth.index.values).agg(
            {'wordcounts': 'sum',
             plot_by: 'first'})
        total_count = growth['wordcounts'].cumsum()
        growth = growth.reset_index().set_index(['index', plot_by]).unstack(
            level=1)
        growth = growth.fillna(0).cumsum()
        growth.columns = growth.columns.droplevel(0)
        growth['total growth'] = total_count
        # Setup the plot
        axes = plt.figure().add_subplot(111)
        plt.style.use(SETTINGS['style'])
        growth.plot(ax=axes, title="Growth of comment threads in {}".format(
            project).title(), fontsize=fontsize)
        axes.set_xlabel("Dates")
        axes.set_ylabel("Cummulative wordcount")
        first, last, *_ = ac.check_date_type(first, last)
        plt.xlim(max(growth.index[0], first), min(growth.index[-1], last))
        ac.show_or_save(show)
