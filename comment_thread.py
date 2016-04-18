
"""
Module that includes the CommentThread parent class,
subclasses for Polymath, Gil Kalai, Gowers, SBSeminar and Terry Tao,
and the MultiCommentThread.
Main libraries used: BeautifullSoup and networkx.
"""

# Imports
from collections import defaultdict
from concurrent import futures
from datetime import datetime
from os import remove
import re
import sys
from urllib.parse import urlparse
import yaml

from bs4 import BeautifulSoup
import joblib
from matplotlib.dates import date2num, DateFormatter, DayLocator, MonthLocator
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
from pandas import DataFrame
import requests


import access_classes as ac
import export_classes as ec

# Loading settings
with open("settings.yaml", "r") as settings_file:
    SETTINGS = yaml.safe_load(settings_file.read())
CMAP = getattr(plt.cm, SETTINGS['cmap'])

with open("author_convert.yaml", "r") as convert_file:
    CONVERT = yaml.safe_load(convert_file.read())

# Pre-declaring dict for selection of subclass of CommentThread
THREAD_TYPES = {}


# Main
def main(project, do_more=True, use_cached=False, cache_it=False):
    """
    Creates threads for all urls of the supplied project, and merges the threads
    into a MultiCommentThread.
    Optionally cashes and re-uses CommentThread instances.
    Optionally tests certain methods of MultiCommentThread
    """
    rec_lim = 8000 if cache_it else 1000
    print("Setting recursionlimit to ", rec_lim)
    sys.setrecursionlimit(rec_lim)

    # loading urls for project
    with open("DATA/" + project.replace(" ", "") + ".csv", "r") as data_input:
        urls = pd.read_csv(data_input, index_col="Ord")['url']

    def create_and_save_thread(enum_url):
        """Returns correct subclass of CommentThread by parsing
        url and calling THREAD_TYPES."""
        enum, url = enum_url
        filename = "CACHE/" + project + "_" + str(enum) + "_thread.p"

        def create_and_save():
            """Helper-function to create and save thread"""
            thread_type = urlparse(url).netloc.split('.')[0].title()
            thread = THREAD_TYPES[thread_type](url)
            if cache_it:
                print("saving {}".format(filename))
                try:
                    joblib.dump(thread, filename)
                    print(filename, " saved")
                except RecursionError:
                    print("Could not pickle ", filename)
                    try:
                        remove(filename)
                        print(filename, " deleted")
                    except IOError:
                        pass
            return thread

        if use_cached:
            try:
                print("loading {}".format(filename))
                thread = joblib.load(filename)
                print(filename, " loaded")
            except (IOError, EOFError) as err:
                print("Could not load ", filename, err)
                try:
                    remove(filename)
                except IOError:
                    pass
                thread = create_and_save()
            return thread
        else:
            try:
                remove(filename)
            except IOError:
                pass
            return create_and_save()

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        the_threads = executor.map(create_and_save_thread, enumerate(urls))
    print("Merging threads in mthread:", end=' ')
    an_mthread = MultiCommentThread(*list(the_threads))
    print("complete")
    if do_more:
        # an_mthread.k_means()
        # return an_mthread
        an_mthread.draw_graph(intervals=50, last='2009-08-31')
        # an_mthread.plot_growth(by='author', last='2009-04-30')
        # an_mthread.plot_activity('thread', intervals=1, last='2009-08-31')
        print("Done")
    else:
        return an_mthread


# Classes
class CommentThread(ac.ThreadAccessMixin, object):
    """
    Parent class for storing a comment thread to a WordPress
    post in a directed graph.
    Subclasses have the appropriate parsing method.
    Inherits methods from parent Mixin.

    Attributes:
        thread_url: parsed url (dict-like)
        _req: request from url (serialized after first time loaded)
        soup: BeautifullSoup parsing of content of request.
        post_title: title of post (only supplied by sub-classes).
        post_content: content of post (only supplied by sub-classes).
        graph: DiGraph based on thread (overruled from ThreadAccessMixin).
        author_name: dict with nodes as keys and authors as values
        authors: set with author-names

    Methods:
        parse_thread: not implemented.
        get_seq_nr: looks for numbers that serve as implicit refs in comments
                    (only called for Polymath 1; result isn't used yet)
        store_attributes: returns arguments in a dict
                          (called by child-classes)
        create_edges: takes graph and returns graph with edges added
                      (called by child classes)
        from ThreadAccessMixin:
            comment_report: takes node-id(s)
                            returns dict with report about node
            print_nodes: takes nodes-id(s),
                         prints out node-data as yaml
            tokenize_and_stem: tokenizes and stems com_content
                               (called as static method by store_attributes)
            strip_proppers_POS: removes propper nouns from comments
                                (currently unused)
    """

    def __init__(self, url, comments_only):
        super(CommentThread, self).__init__()
        self.thread_url = urlparse(url)
        reqfile = 'CACHED_DATA/' + \
            self.thread_url.netloc.split('.')[0] + \
            ('_').join(self.thread_url.path.split('/')[:-1]) + '_req.p'
        try:
            self._req = joblib.load(reqfile)
        except IOError:
            try:
                self._req = requests.get(url)
            except (requests.exceptions.ConnectionError) as err:
                print("Could not connect: {}".format(err))
                sys.exit(1)
            joblib.dump(self._req, reqfile)
        # faster parsers do not work
        self.soup = BeautifulSoup(self._req.content, SETTINGS['parser'])
        self.graph = self.parse_thread(self.soup, self.thread_url)
        self.post_title = ""
        self.post_content = ""
        # creates sub_graph and node:author dict based on comments_only
        if comments_only:
            # create node:name dict for nodes that are comments
            self.node_name = {node_id: data['com_author'] for (node_id, data)
                              in self.graph.nodes_iter(data=True) if
                              data['com_type'] == 'comment'}
            # create sub_graph based on keys of node_name
            self.graph = self.graph.subgraph(self.node_name.keys())
        else:
            self.node_name = nx.get_node_attributes(self.graph, 'com_author')
        self.authors = set(self.node_name.values())

    @classmethod
    def parse_thread(cls, a_soup, url):
        """Abstract method: raises NotImplementedError."""
        raise NotImplementedError("Subclasses should implement this!")

    @classmethod
    def get_seq_nr(cls, content, url):
        """Looks for numbers in comments (implicit refs)"""
        if url.path.split("/")[-2] not in SETTINGS["implicit_refs"]:
            return None
        else:
            pattern = re.compile(r"\(\d+\)|\d+.\d*")
            content = "\n".join(content)
            matches = pattern.findall(content)
            try:
                seq_nr = matches[0]
                if seq_nr.startswith('('):
                    seq_nr = [int(seq_nr.strip("(|)"))]
                elif "." in seq_nr:
                    seq_nr = [int(i) for i in seq_nr.split(".") if i]
                else:
                    seq_nr = None
            except IndexError:
                seq_nr = None
            return seq_nr

    @classmethod
    def store_attributes(cls,
                         com_class, com_depth, com_all_content, time_stamp,
                         com_author, com_author_url, child_comments,
                         thread_url, seq_nr):
        """Processes post-content, and returns arguments as dict"""
        content = " ".join(com_all_content)
        tokens, stems = ac.ThreadAccessMixin.tokenize_and_stem(content)
        return {"com_type": com_class[0],
                "com_depth": com_depth,
                "com_content": content,
                "com_tokens": tokens,
                "com_stems": stems,
                "com_timestamp": time_stamp,
                "com_author": com_author,
                "com_author_url": com_author_url,
                "com_children": child_comments,
                "com_thread": thread_url,
                "com_seq_nr": seq_nr}

    @classmethod
    def create_edges(cls, a_graph):
        """
        Takes nx.DiGraph, adds edges to child_comments and returns nx.DiGraph.
        """
        for node_id, children in\
                nx.get_node_attributes(a_graph, "com_children").items():
            if children:
                a_graph.add_edges_from(((child,
                                         node_id) for child in children))
        return a_graph


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
        thread_urls: list of urls of the respective threads
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
        plot_activity: accessor method plotting of x-axis: time_stamps
                                                   y-axis: what's active
        plot_growth: accessor method plotting of x-axis:  time_stamps
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
        self.thread_urls = []
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
            print("Overlapping threads found when adding {}".format(
                thread.post_title))
            print("Overlapping nodes: {}".format(overlap))
        self.node_name.update(thread.node_name)
        self.thread_urls.append(thread.thread_url)
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

    # Accessor methods
    def draw_graph(self, intervals=10,
                   first=SETTINGS['first_date'],
                   last=SETTINGS['last_date'],
                   show=True, project=SETTINGS['msg']):
        """Draws and shows (alt: saves) DiGraph of MultiCommentThread as tree-structure.
        Should be called with project as kwarg for correct title."""
        # creating title and axes
        figure = plt.figure()
        figure.suptitle("Thread structure for {}".format(project).title(),
                        fontsize=12)
        axes = figure.add_subplot(111)
        axes.yaxis.set_major_locator(DayLocator(interval=intervals))
        axes.yaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_ticks(list(range(1, 7)))
        axes.set_xlabel("Comment Levels")
        try:
            first = first if isinstance(
                first, datetime) else datetime.strptime(
                    first, "%Y-%m-%d")
            last = last if isinstance(last, datetime) else datetime.strptime(
                last, "%Y-%m-%d")
        except ValueError as err:
            print(err, ": datetime failed")
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
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_activity(self, activity,
                      first=SETTINGS['first_date'],
                      last=SETTINGS['last_date'],
                      intervals=1,
                      show=True,
                      project=SETTINGS['msg']):
        """
        Plots and shows (alt: saves) plot of
            x-axis: time_stamps,
            y-axis: what's active (author / thread).
        Set project as kwarg for correct title
        """
        stop = datetime(2000, 1, 1)
        start = datetime.now()
        if activity.lower() == "author":
            items = list(self.author_color.keys())
            tick_tuple = tuple(items)
            key = "com_author"
        elif activity.lower() == "thread":
            items = self.thread_urls
            tick_tuple = tuple([item.netloc + "\n" + item.path for
                                item in items])
            key = "com_thread"
        else:
            pass
        for y_value, item in enumerate(items, start=1):
            norm = mpl.colors.Normalize(vmin=SETTINGS['vmin'],
                                        vmax=SETTINGS['vmax'])
            c_mp = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
            v_color = c_mp.to_rgba(self.author_color[item]) if\
                activity.lower() == "author" else 'k'
            timestamps = [data["com_timestamp"] for _, data in
                          self.graph.nodes_iter(data=True)
                          if data[key] == item]
            this_start, this_stop = min(timestamps), max(timestamps)
            start, stop = min(start, this_start), max(stop, this_stop)
            plt.hlines(y_value, this_start, this_stop, v_color, lw=.5)
            for timestamp in timestamps:
                plt.vlines(timestamp,
                           y_value+0.05, y_value-0.05,
                           v_color, lw=1)
        # Setup the plot
        plt.title("{} activity over time for {}".format(
            activity, project).title(), fontsize=12)
        plt.style.use(SETTINGS['style'])
        axes = plt.gca()
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_major_locator(MonthLocator(interval=intervals))
        fontsize = 4 if len(items) >= 15 else 6
        axes.set_yticklabels(items, fontsize=fontsize)
        try:
            first = first if isinstance(
                first, datetime) else datetime.strptime(
                    first, "%Y-%m-%d")
            last = last if isinstance(last, datetime) else datetime.strptime(
                last, "%Y-%m-%d")
        except ValueError as err:
            print(err, ": datetime failed")
        plt.xlim(max([start, first]),
                 min([stop, last]))
        plt.yticks(range(1, len(items)+1), tick_tuple)
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_growth(self, by='thread_type',
                    first=SETTINGS['first_date'], last=SETTINGS['last_date'],
                    show=True,
                    project=SETTINGS['msg']):
        """Plots and shows (alt: saves) how fast a thread grows (cumsum of wordcounts)
        Set project as kwarg for correct title"""
        if by == 'thread_type':
            stamps, thread_types, wordcounts = zip(
                *((data["com_timestamp"],
                   data["com_thread"].netloc.split('.')[0],
                   len(data["com_tokens"]))
                  for _, data in self.graph.nodes_iter(data=True)))
        elif by == 'thread':
            stamps, thread_types, wordcounts = zip(
                *((data["com_timestamp"],
                   data["com_thread"].path.split('/')[-2],
                   len(data["com_tokens"]))
                  for _, data in self.graph.nodes_iter(data=True)))
        elif by == 'author':
            stamps, thread_types, wordcounts = zip(
                *((data["com_timestamp"],
                   data["com_author"],
                   len(data["com_tokens"]))
                  for _, data in self.graph.nodes_iter(data=True)))
        else:
            raise ValueError("By is either thread_type of thread")
        growth = DataFrame(
            {'wordcounts': wordcounts, 'thread type': thread_types},
            index=stamps)
        growth.sort_index(inplace=True)
        assert len(thread_types) != 0
        for thread_type in set(thread_types):
            this_growth = growth[
                growth['thread type'] == thread_type]['wordcounts']
            this_growth = DataFrame({thread_type: this_growth.cumsum()})
            growth = pd.merge(growth, this_growth,
                              left_index=True, right_index=True,
                              how='left')
            growth[thread_type] = growth[thread_type].fillna(
                method='ffill').fillna(0)
        growth['total growth'] = growth['wordcounts'].cumsum()
        growth.drop(['thread type', 'wordcounts'], inplace=True, axis=1)
        # Setup the plot
        axes = plt.figure().add_subplot(111)
        plt.style.use(SETTINGS['style'])
        growth.plot(ax=axes, title="Growth of comment threads in {}".format(
            project).title())
        axes.set_ylabel("Cummulative wordcount")
        try:
            first = first if isinstance(
                first, datetime) else datetime.strptime(
                    first, "%Y-%m-%d")
            last = last if isinstance(last, datetime) else datetime.strptime(
                last, "%Y-%m-%d")
        except ValueError as err:
            print(err, ": datetime failed")
        plt.xlim(max(growth.index[0], first), min(growth.index[-1], last))
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath Blog, with method for actual parsing. """
    def __init__(self, url, comments_only=True):
        super(CommentThreadPolymath, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post"}).find("h3").text
        self.post_content = self.soup.find(
            "div", {"class": "storycontent"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"id": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word
                             in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find(
                                   "div", {"class": "comment-author vcard"}
                                   ).find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").find("span").text
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp (and time is poped from all_content)
            time_stamp = " ".join(com_all_content.pop().split()[-7:])[2:]
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%B %d, %Y @ %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find(
                    "a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # get sequence-number of comment (if available)
            seq_nr = cls.get_seq_nr(com_all_content, thread_url)
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find(
                    "ul", {"class": "children"}).find_all(
                        "li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url,
                                        seq_nr)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadGilkalai(CommentThread):
    """ Child class for Gil Kalai Blog, with method for actual parsing. """
    def __init__(self, url, comments_only=True):
        super(CommentThreadGilkalai, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"id": "content"}).find("h2", {"class": "entry-title"}).text
        self.post_content = self.soup.find(
            "div", {"class": "entry-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"class": "commentlist"})
        if the_comments:
            # NOTE: Pingbacks have no id and are ignored
            all_comments = the_comments.find_all("li", {"class": "comment"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.find("div").get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if
                             word.startswith("depth-"))
            com_all_content = [item.text for item in comment.find(
                "div", {"class": "comment-body"}).find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").text.strip()
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find(
                "div", {"class": "comment-meta commentmetadata"}).text.strip()
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%B %d, %Y at %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find(
                    "a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # get sequence-number of comment (if available)
            seq_nr = cls.get_seq_nr(com_all_content, thread_url)
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find(
                    "ul", {"class": "children"}).find_all(
                        "li", {"class": depth_search})
                child_comments = [child.find("div").get("id") for
                                  child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url,
                                        seq_nr)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadGowers(CommentThread):
    """ Child class for Gowers Blog, with method for actual parsing."""
    def __init__(self, url, comments_only=True):
        super(CommentThreadGowers, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post"}).find("h2").text
        self.post_content = self.soup.find(
            "div", {"class": "post"}).find(
                "div", {"class": "entry"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"class": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for
                             word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").text.strip()
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find("small").find("a").text
            try:
                time_stamp = datetime.strptime(
                    time_stamp, "%B %d, %Y at %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find(
                    "a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # get sequence-number of comment (if available)
            seq_nr = cls.get_seq_nr(com_all_content, thread_url)
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find(
                    "ul", {"class": "children"}).find_all(
                        "li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url,
                                        seq_nr)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadSBSeminar(CommentThread):
    """
    Child class for Secret Blogging Seminar, with method for actual parsing.
    """
    def __init__(self, url, comments_only=True):
        super(CommentThreadSBSeminar, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "article").find("h1", {"class": "entry-title"}).text
        self.post_content = self.soup.find(
            "div", {"class": "entry-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx_DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"class": "comment-list"})
        if the_comments:
            all_comments = the_comments.find_all("li", {"class": "comment"})
        else:
            all_comments = []
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if
                             word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find(
                                   "div", {"class": "comment-content"}
                                   ).find_all("p")]
            # getting and converting author_name and getting url
            com_author_and_url = comment.find(
                "div", {"class": "comment-author"}
                ).find("cite", {"class": "fn"})
            try:
                com_author = com_author_and_url.find("a").text
                com_author_url = com_author_and_url.find("a").get("href")
            except AttributeError:
                try:
                    com_author = com_author_and_url.text
                    com_author_url = None
                except AttributeError as err:
                    print(err, com_author_and_url)
                    com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find(
                "div", {"class": "comment-metadata"}).find("time").get(
                    "datetime")
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%Y-%m-%dT%H:%M:%S+00:00")
            except ValueError as err:
                print(err, ": datetime failed for {}".format(time_stamp))
            # get sequence-number of comment (if available)
            seq_nr = cls.get_seq_nr(com_all_content, thread_url)
            # make list of child-comments (only id's) VOID IN THIS CASE
            child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url,
                                        seq_nr)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges VOID
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadTerrytao(CommentThread):
    """ Child class for Tao Blog, with method for actual parsing."""
    def __init__(self, url, comments_only=True):
        super(CommentThreadTerrytao, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post-meta"}).find("h1").text
        self.post_content = self.soup.find(
            "div", {"class": "post-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("div", {"class": "commentlist"})
        if the_comments:
            # this seems to ignore the pingbacks
            all_comments = the_comments.find_all("div", {"class": "comment"})
            all_comments += the_comments.find_all("div", {"class": "pingback"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if
                             word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find_all("p")][2:]
            # getting and converting author_name
            try:
                com_author = comment.find(
                    "p", {"class": "comment-author"}).text
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find(
                "p", {"class": "comment-permalink"}).find("a").text
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%d %B, %Y at %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find(
                    "p", {"class": "comment-author"}).find("a").get("href")
            except AttributeError:
                com_author_url = None
            # get sequence-number of comment (if available)
            seq_nr = cls.get_seq_nr(com_all_content, thread_url)
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                if comment.next_sibling.next_sibling['class'] == ['children']:
                    child_comments = comment.next_sibling.next_sibling.\
                                    find_all("div",
                                             {"class": "comment"})
                    child_comments = [child.get("id") for child in
                                      child_comments if depth_search in
                                      child["class"]]
                else:
                    child_comments = []
            except (AttributeError, TypeError):
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url,
                                        seq_nr)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph

THREAD_TYPES = {"Polymathprojects": CommentThreadPolymath,
                "Gilkalai": CommentThreadGilkalai,
                "Gowers": CommentThreadGowers,
                "Sbseminar": CommentThreadSBSeminar,
                "Terrytao": CommentThreadTerrytao}

if __name__ == '__main__':
    ARGUMENT = sys.argv[1:]
    if ARGUMENT:
        SETTINGS['msg'] = input("Message to be used: ")
        main(ARGUMENT, use_cached=True, cache_it=True)
    else:
        print(SETTINGS['msg'])
        main(SETTINGS['project'], do_more=False,
             use_cached=True, cache_it=True)
