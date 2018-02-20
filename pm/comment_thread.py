
"""
Module that includes the CommentThread parent class,
subclasses for Polymath, Gil Kalai, Gowers, SBSeminar and Terry Tao.
Main libraries used: BeautifullSoup and networkx.
"""

# Imports
from bisect import insort
from collections import OrderedDict
from concurrent import futures
import datetime
import logging
from operator import methodcaller
from os import path
import sys
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import joblib
import networkx as nx
import pandas as pd
import requests


import access_classes as ac
import comment as cm
from cluster_helper import ClusterNodes
import multi_comment_thread as mc

# Loading settings
SETTINGS, CMAP, LOCATION = ac.load_settings()
CONVERT, LASTS, *_ = ac.load_yaml("settings/author_convert.yaml",
                                  "settings/lasts_by_date.yaml")


# Pre-declaring dict for selection of subclass of CommentThread
THREAD_TYPES = {}
# actions to be used as argument for --more
ACTIONS = {"growth": "plot_growth",
           "growth_size": "plot_growth_size",
           "wordcounts": "plot_sizes",
           "author_activity": "plot_activity_author",
           "thread_activity": "plot_activity_thread"}
# recursion-limits
REC_LIMIT_STND = sys.getrecursionlimit()
REC_LIMIT_HIGH = 10000


# Main
def main(project, **kwargs):
    """
    Creates threads for all urls of the supplied project.
    Optionally cashes and re-uses CommentThread instances.
    Tests certain methods of MultiCommentThread or returns:
        MultiCommentThread from merged Threads, or
        OrderedDict of unmerged CommentThreads.
    """
    cache_it = kwargs.get('cache_it', False)
    do_more = kwargs.get('do_more', False)
    if cache_it:
        sys.setrecursionlimit(REC_LIMIT_HIGH)

    # loading urls for project
    data_file = path.join(LOCATION,
                          "DATA/" + project.replace(" ", "") + ".csv")
    with open(data_file, "r") as data_input:
        data = pd.read_csv(data_input, index_col="Ord", encoding='utf-8')
        urls = data['url']
        is_research = data['research']

    def create_and_save_thread(enum_url):
        """Returns correct subclass of CommentThread by parsing
        url and calling THREAD_TYPES."""
        enum, (url, is_research) = enum_url
        filename = "CACHE/" + project + "_" + str(enum) + "_thread.p"

        def create_and_process():
            """Helper-function to create , optionally save,
            and return thread"""
            thread_url = urlparse(url)
            thread_type = thread_url.netloc.split('.')[0].title()
            thread = THREAD_TYPES[thread_type](url, is_research)
            request_file = "CACHED_DATA/" + \
                           thread_url.netloc.split('.')[0] + \
                           ('_').join(
                               thread_url.path.split('/')[:-1]) + '_req.p'
            request_file = path.join(LOCATION, request_file)
            if kwargs.get('delete_all', False):
                ac.handle_delete(filename)
                ac.handle_delete(request_file)
            if cache_it:
                ac.to_pickle(thread, filename)
            return thread

        if kwargs.get('use_cached', False):
            try:
                thread = joblib.load(filename)
            except (IOError, EOFError) as err:
                logging.warning("Could not load %s: %s", filename, err)
                ac.handle_delete(filename)
                thread = create_and_process()
        else:  # not used_cached
            ac.handle_delete(filename)
            thread = create_and_process()
        assert isinstance(thread, CommentThread)
        return thread

    if not cache_it:  # multi-threading of creation of threads
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            the_threads = executor.map(
                create_and_save_thread, enumerate(zip(urls, is_research)))
    else:  # cache_it without multi-threading
        the_threads = (create_and_save_thread(enum_url) for
                       enum_url in enumerate(zip(urls, is_research)))
    if not kwargs.get('merge', True):
        if do_more:
            logging.warning("Do more overridden by no-merge")
        title_thread = OrderedDict(
            ((thread.post_title, thread) for thread in the_threads))
        try:
            titles1 = [string.replace('\xa0', ' ') for string in
                       title_thread.keys()]
            titles2 = [string.replace('\xa0', ' ') for string in
                       data['title']]
            assert titles1 == titles2
        except AssertionError:
            logging.warning("Threads not in proper order")
            for thread1, thread2 in zip(titles1, titles2):
                if thread1 != thread2:
                    print(thread1)
                    print(thread2)
                    print()
        except TypeError:
            logging.warning("Casting to list or comparison failed")
        except ValueError:
            logging.warning("Replacement of chars failed")
        return title_thread
    else:
        the_threads = list(the_threads)
        logging.info("Merging threads in mthread")
        an_mthread = mc.MultiCommentThread(*the_threads)
        logging.info("Merging completed")
    if do_more:
        the_project = project.replace("pm", "Polymath ")\
            if project.startswith("pm")\
            else project.replace("mini_pm", "Mini-Polymath ")
        do_this = methodcaller(ACTIONS[do_more], project=the_project)
        do_this(an_mthread)
        logging.info("Processing complete at %s",
                     datetime.datetime.now().strftime("%H:%M:%S"))
    else:
        return an_mthread


# Classes
class ThreadData(object):
    """Class with url and soup for thread"""

    def __init__(self, url, is_research):
        self.is_research = is_research
        self.url = url
        self.thread_url = urlparse(url)
        self._make_request()
        # faster parsers do not work
        self.soup = BeautifulSoup(self._req.content.decode(
            'utf-8', 'ignore'), SETTINGS['parser'])

    def _make_request(self):
        reqfile = 'CACHED_DATA/' + \
            self.thread_url.netloc.split('.')[0] + \
            ('_').join(self.thread_url.path.split('/')[:-1]) + '_req.p'
        reqfile = path.join(LOCATION, reqfile)
        try:
            self._req = joblib.load(reqfile)
        except IOError:
            try:
                self._req = requests.get(self.url)
            except (requests.exceptions.ConnectionError) as err:
                logging.exception("Could not connect: %s", err)
                sys.exit(1)
            else:
                joblib.dump(self._req, reqfile)


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
        cluster_comments: takes graph and clusters comments based on
                          timestamps.
                          returns graph with cluster-ids as node-attributes.
        from ThreadAccessMixin:
            comment_report: takes node-id(s)
                            returns dict with report about node
            print_nodes: takes nodes-id(s),
                         prints out node-data as yaml
    """

    def __init__(self, url, is_research, comments_only):
        super(CommentThread, self).__init__()
        self.data = ThreadData(url, is_research)
        self.timestamps = []
        self.parse_thread()
        self.post_title = ""
        self.post_content = ""
        # creates sub_graph and node:author dict based on comments_only
        if comments_only:
            # create node:name dict for nodes that are comments
            self.node_name = {node_id: data['com_author'] for (node_id, data)
                              in self.graph.nodes(data=True) if
                              data['com_type'] == 'comment'}
            # create sub_graph based on keys of node_name
            self.graph = self.graph.subgraph(self.node_name.keys())
        else:
            self.node_name = nx.get_node_attributes(self.graph, 'com_author')
        self.authors = set(self.node_name.values())

    def parse_thread(self):
        """Abstract method: raises NotImplementedError."""
        raise NotImplementedError("Subclasses should implement this!")

    def process_comment(self, comment):
        """Abstract method: raises NotImplementedError."""
        raise NotImplementedError("Subclasses should implement this!")

    def parse_thread_generic(self, fun1, fun2):
        """Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init."""
        self.graph = nx.DiGraph()
        the_comments = fun1(self.data.soup)
        if the_comments:
            all_comments = fun2(the_comments)
            if self.__class__ == CommentThreadTerrytao:
                all_comments += the_comments.find_all(
                    "div", {"class": "pingback"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            self.process_comment(comment)
        self.create_edges()
        self.remove_comments()

    def record_timestamp(self, timestamp):
        """adds timestamp to sorted list of timestamps"""
        insort(self.timestamps, timestamp)

    def create_node(self, comment_data):
        """adds node for com_id and attributes from node_attr to self.graph"""
        com_id, node_attr = comment_data()
        self.graph.add_node(com_id, **node_attr)
        logging.debug("Created %s", com_id)
        self.record_timestamp(node_attr['com_timestamp'])

    def create_edges(self):
        """
        Takes nx.DiGraph, adds edges to child_comments and returns nx.DiGraph.
        """
        for node_id, children in\
                nx.get_node_attributes(self.graph, "com_children").items():
            if children:
                self.graph.add_edges_from(((child, node_id) for
                                           child in children))

    def remove_comments(self):
        """Lookups last to be included comments in LASTS and
        removes all later comments."""
        try:
            last_date = LASTS[self.data.url]
        except KeyError:
            logging.warning("Moving on without removing comments for %s",
                            self.data.url)
            return
        else:
            last_datetime = datetime.datetime(last_date.year,
                                              last_date.month,
                                              last_date.day)
        logging.debug("Removing from %s, in %s", last_date, self.data.url)
        to_remove = [node for (node, date) in nx.get_node_attributes(
            self.graph, "com_timestamp").items() if date > last_datetime]
        if to_remove:
            logging.debug("Removing comments %s", to_remove)
            self.graph.remove_nodes_from(to_remove)

    def cluster_comments(self):
        """
        Clusters comments based on their timestamps and
        assigns cluster-membership as attribute to nodes.
        """
        data = ClusterNodes(self.graph, self.post_title).data_
        for com_id in data.index:
            try:
                self.graph.node[com_id]['cluster_id'] = (
                    data.loc[
                        com_id, 'cluster_id'],
                    data.loc[com_id, 'weight'],
                    data.loc[com_id, 'author_weight'])
            except KeyError as err:
                print(err)
                print(data.columns)


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath Blog, with method for actual parsing. """

    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadPolymath, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.data.soup.find(
            "div", {"class": "post"}).find("h3").text.strip()
        self.post_content = self.data.soup.find(
            "div", {"class": "storycontent"}).get_text()
        self.cluster_comments()

    def parse_thread(self):
        """
        Supplies parameters for Polymath-blog
        to call parse_thread of superclass.
        """
        self.parse_thread_generic(
            methodcaller("find", "ol", {"id": "commentlist"}),
            methodcaller("find_all", "li"))

    def process_comment(self, comment):
        """Processes soup from single comment of Polymath blog
        and creates node with corresponding attributes."""
        # identify id, class, depth and content
        comment_data = cm.Comment(cm.PolymathCommentParser,
                                  comment, self.data.thread_url)
        self.create_node(comment_data)


class CommentThreadGilkalai(CommentThread):
    """ Child class for Gil Kalai Blog, with method for actual parsing. """

    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadGilkalai, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.data.soup.find(
            "div", {"id": "content"}).find(
                "h2", {"class": "entry-title"}).text.strip()
        self.post_content = self.data.soup.find(
            "div", {"class": "entry-content"}).get_text()
        self.cluster_comments()

    def parse_thread(self):
        """
        Supplies parameters for Gil Kalai blog
        to call parse_thread of superclass.
        """
        self.parse_thread_generic(
            methodcaller("find", "ol", {"class": "commentlist"}),
            methodcaller("find_all", "li", {"class": "comment"}))

    def process_comment(self, comment):
        """Processes soup from single comment of Gil Kalai blog
        and creates node with corresponding attributes."""
        comment_data = cm.Comment(cm.GilkalaiCommentParser,
                                  comment, self.data.thread_url)
        self.create_node(comment_data)


class CommentThreadGowers(CommentThread):
    """ Child class for Gowers Blog, with method for actual parsing."""

    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadGowers, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.data.soup.find(
            "div", {"class": "post"}).find("h2").text.strip()
        self.post_content = self.data.soup.find(
            "div", {"class": "post"}).find(
                "div", {"class": "entry"}).get_text()
        self.cluster_comments()

    def parse_thread(self):
        """
        Supplies parameters for Gower blog
        to call parse_thread of superclass.
        """
        self.parse_thread_generic(
            methodcaller("find", "ol", {"class": "commentlist"}),
            methodcaller("find_all", "li"))

    def process_comment(self, comment):
        """Processes soup from single comment of Gowers blog
        and creates node with corresponding attributes."""
        # identify id, class, depth and content
        comment_data = cm.Comment(cm.GowersCommentParser,
                                  comment, self.data.thread_url)
        self.create_node(comment_data)


class CommentThreadSBSeminar(CommentThread):
    """
    Child class for Secret Blogging Seminar, with method for actual parsing.
    """

    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadSBSeminar, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.data.soup.find(
            "article").find("h1", {"class": "entry-title"}).text.strip()
        self.post_content = self.data.soup.find(
            "div", {"class": "entry-content"}).get_text()
        self.cluster_comments()

    def parse_thread(self):
        """
        Supplies parameters for SBS blog
        to call parse_thread of superclass.
        """
        self.parse_thread_generic(
            methodcaller("find", "ol", {"class": "comment-list"}),
            methodcaller("find_all", "li", {"class": "comment"}))

    def process_comment(self, comment):
        """Processes soup from single comment of SBS blog
        and creates node with corresponding attributes."""
        comment_data = cm.Comment(cm.SBSCommentParser,
                                  comment, self.data.thread_url)
        self.create_node(comment_data)


class CommentThreadTerrytao(CommentThread):
    """ Child class for Tao Blog, with method for actual parsing."""

    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadTerrytao, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.data.soup.find(
            "div", {"class": "post-meta"}).find("h1").text.strip()
        self.post_content = self.data.soup.find(
            "div", {"class": "post-content"}).get_text()
        self.cluster_comments()

    def parse_thread(self):
        """
        Supplies parameters for Tao blog
        to call parse_thread of superclass.
        """
        self.parse_thread_generic(
            methodcaller("find", "div", {"class": "commentlist"}),
            methodcaller("find_all", "div", {"class": "comment"}))

    def process_comment(self, comment):
        """Processes soup from single comment of Tao blog
        and creates node with corresponding attributes."""
        comment_data = cm.Comment(cm.TaoCommentParser,
                                  comment, self.data.thread_url)
        self.create_node(comment_data)


THREAD_TYPES = {"Polymathprojects": CommentThreadPolymath,
                "Gilkalai": CommentThreadGilkalai,
                "Gowers": CommentThreadGowers,
                "Sbseminar": CommentThreadSBSeminar,
                "Terrytao": CommentThreadTerrytao}


if __name__ == '__main__':
    PARSER = ac.make_arg_parser(
        ACTIONS.keys(), SETTINGS['project'],
        "Process the threads of a given project")
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
