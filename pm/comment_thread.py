
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
import re
import sys
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import requests
from sklearn.cluster import DBSCAN


import access_classes as ac
import comment as cm
import multi_comment_thread as mc
import text_functions as tf

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

    @staticmethod
    def get_seq_nr(content, url):
        """Looks for numbers in comments (implicit refs)"""
        if url.path.split("/")[-2] not in SETTINGS["implicit_refs"]:
            seq_nr = None
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

    def record_timestamp(self, timestamp):
        """adds timestamp to sorted list of timestamps"""
        insort(self.timestamps, timestamp)

    def create_node(self, com_id, node_attr):
        """adds node for com_id and attributes from node_attr to self.graph"""
        expected = ['com_type', 'com_depth', 'com_author', 'com_timestamp',
                    'com_content', 'com_author_url', 'com_children',
                    'com_thread']
        if SETTINGS['find implicit references']:
            expected.append('seq_nr')
        try:
            assert all(key in node_attr for key in expected)
        except AssertionError:
            logging.warning("Missing attributes for, %s", com_id)
        node_attr['com_content'] = " ".join(node_attr['com_content'])
        node_attr['com_tokens'] = tf.tokenize(node_attr['com_content'])
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

    @staticmethod
    def get_conv_author(comment, parse_fun):
        """Parses comment to find author, and converts to avoid duplicates"""
        try:
            com_author = parse_fun(comment)
        except AttributeError as err:
            logging.warning("%s, %s", err, comment)
            com_author = "Unable to resolve"
        if com_author in CONVERT:
            com_author = CONVERT[com_author]
        else:  # no redundant spaces when converted
            com_author = re.sub(' +', ' ', com_author.strip())
        return com_author

    @staticmethod
    def parse_timestamp(time_stamp, date_format):
        """Parses time_stamp to datetime object."""
        try:
            time_stamp = datetime.datetime.strptime(
                time_stamp, date_format)
        except ValueError as err:
            logging.warning("%s: datetime failed", err)
            print(time_stamp)
        return time_stamp

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

    def __handle_cluster_data(self):
        """Helper-function that iterates over network,
        and returns DataFrame of timstamp-data of all comments."""
        node_data = dict((node, {'timestamps': data['com_timestamp'],
                                 'authors': data['com_author']})
                         for node, data in self.graph.nodes(data=True))
        try:
            data = DataFrame(node_data).T.sort_values('timestamps')
        except KeyError:
            print(node_data)
        for node in data.index:
            try:
                assert data.loc[node, 'timestamps'] == self.graph.node[
                    node]['com_timestamp']
            except AssertionError:
                print("Mismatch for ", node)
        epoch = data.ix[0, 'timestamps']
        data['timestamps'] = (data['timestamps'] - epoch).astype(int)
        return data

    @staticmethod
    def __cluster_timestamps(data, post_title):
        cluster_data = data['timestamps'].as_matrix().reshape(-1, 1)
        one_day = 86400000000000  # timedelta(1) to int to match data
        times_db = DBSCAN(eps=one_day / 2,
                          min_samples=2,
                          metric="euclidean")
        labels = times_db.fit_predict(cluster_data)
        unique_labels = np.sort(np.unique(labels))
        logging.info("Found %i clusters in %s",
                     len(unique_labels) - 1, post_title)
        try:
            assert len(labels) == len(cluster_data)
        except AssertionError as err:
            logging.warning("Mismatch cluster-labels: %s", err)
            print(unique_labels)
            print(labels)
        data['cluster_id'] = labels
        return data

    def cluster_comments(self):
        """
        Clusters comments based on their timestamps and
        assigns cluster-membership as attribute to nodes.
        """
        data = self.__handle_cluster_data()
        data = self.__cluster_timestamps(data, self.post_title)
        comments_per_cluster = data['cluster_id'].value_counts()
        data['weight'] = [comments_per_cluster[cluster] for
                          cluster in data['cluster_id']]
        a_weights = data.groupby(
            ['cluster_id', 'authors']).count()['timestamps']
        data['author_weight'] = data.apply(
            lambda x: a_weights[x['cluster_id'], x['authors']], axis=1)
        data = data.replace(to_replace={'cluster_id': {-1: np.nan}})
        data.loc[np.isnan(data['cluster_id']), 'weight'] = 0
        data.loc[np.isnan(data['cluster_id']), 'author_weight'] = 0
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
        comment_data = cm.CommentPolymath(comment, self.data.thread_url)
        self.create_node(comment_data.com_id, comment_data.node_attr)


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
        # identify id, class, depth and content
        node_attr = {}
        com_id = comment.find("div").get("id")
        com_class = comment.get("class")
        node_attr['com_type'] = com_class[0]
        node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))
        node_attr['com_content'] = [item.text for item in comment.find(
            "div", {"class": "comment-body"}).find_all("p")]
        # getting and converting author_name
        node_attr['com_author'] = self.get_conv_author(
            comment,
            lambda comment: comment.find("cite").text.strip())
        # creating timeStamp
        time_stamp = comment.find(
            "div", {"class": "comment-meta commentmetadata"}).text.strip()
        node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y at %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            node_attr['com_author_url'] = comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          node_attr['com_author'])
            node_attr['com_author_url'] = None
        # get sequence-number of comment (if available)
        if SETTINGS['find implicit references']:
            node_attr['seq_nr'] = self.get_seq_nr(
                node_attr['com_content'], self.data.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(node_attr['com_depth'] + 1)
            child_comments = comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            node_attr['com_children'] = [
                child.find("div").get("id") for child in child_comments]
        except AttributeError:
            node_attr['com_children'] = []
        # adding thread_url
        node_attr['com_thread'] = self.data.thread_url
        self.create_node(com_id, node_attr)


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
        node_attr = {}
        com_id = comment.get("id")
        com_class = comment.get("class")
        node_attr['com_type'] = com_class[0]
        node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))
        node_attr['com_content'] = [
            item.text for item in comment.find_all("p")]
        # getting and converting author_name
        node_attr['com_author'] = self.get_conv_author(
            comment,
            lambda comment: comment.find("cite").text.strip())
        # creating timeStamp
        time_stamp = comment.find("small").find("a").text
        node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%B %d, %Y at %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            node_attr['com_author_url'] = comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          node_attr['com_author'])
            node_attr['com_author_url'] = None
        # get sequence-number of comment (if available)
        node_attr['seq_nr'] = self.get_seq_nr(node_attr['com_content'],
                                              self.data.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(node_attr['com_depth'] + 1)
            child_comments = comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            node_attr['com_children'] = [
                child.get("id") for child in child_comments]
        except AttributeError:
            node_attr['com_children'] = []
        # adding thread_url
        node_attr['com_thread'] = self.data.thread_url
        self.create_node(com_id, node_attr)


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
        # identify id, class, depth and content
        node_attr = {}
        com_id = comment.get("id")
        com_class = comment.get("class")
        node_attr['com_type'] = com_class[0]
        node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))
        node_attr['com_content'] = [
            item.text for item in comment.find(
                "div", {"class": "comment-content"}).find_all("p")]
        # getting and converting author_name and getting url
        com_author_and_url = comment.find(
            "div", {"class": "comment-author"}).find(
                "cite", {"class": "fn"})
        try:
            com_author = com_author_and_url.find("a").text
            node_attr['com_author_url'] = com_author_and_url.find("a").get(
                "href")
        except AttributeError:
            try:
                com_author = com_author_and_url.text
                node_attr['com_author_url'] = None
            except AttributeError:
                logging.debug("Could not resolve author_url for %s",
                              com_author)
                com_author = "unable to resolve"
        node_attr['com_author'] = CONVERT[com_author] if\
            com_author in CONVERT else com_author
        # creating timeStamp
        time_stamp = comment.find(
            "div", {"class": "comment-metadata"}).find("time").get(
                "datetime")
        node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%Y-%m-%dT%H:%M:%S+00:00")
        # get sequence-number of comment (if available)
        node_attr['seq_nr'] = self.get_seq_nr(node_attr['com_content'],
                                              self.data.thread_url)
        # make list of child-comments (only id's) VOID IN THIS CASE
        node_attr['com_children'] = []
        # adding thread_url
        node_attr['com_thread'] = self.data.thread_url
        self.create_node(com_id, node_attr)


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
        # identify id, class, depth and content
        node_attr = {}
        com_id = comment.get("id")
        com_class = comment.get("class")
        node_attr['com_type'] = com_class[0]
        node_attr['com_depth'] = next(
            int(word[6:]) for word in com_class if word.startswith("depth-"))
        node_attr['com_content'] = [
            item.text for item in comment.find_all("p")][2:]
        # getting and converting author_name
        node_attr['com_author'] = self.get_conv_author(
            comment,
            lambda comment: comment.find(
                "p", {"class": "comment-author"}).text)
        # creating timeStamp
        time_stamp = comment.find(
            "p", {"class": "comment-permalink"}).find("a").text
        node_attr['com_timestamp'] = self.parse_timestamp(
            time_stamp, "%d %B, %Y at %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            node_attr['com_author_url'] = comment.find(
                "p", {"class": "comment-author"}).find("a").get("href")
        except AttributeError:
            logging.debug(
                "Could not resolve author_url for %s", node_attr['com_author'])
            node_attr['com_author_url'] = None
        # get sequence-number of comment (if available)
        node_attr['seq_nr'] = self.get_seq_nr(node_attr['com_content'],
                                              self.data.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(node_attr['com_depth'] + 1)
            if comment.next_sibling.next_sibling['class'] == ['children']:
                child_comments = comment.next_sibling.next_sibling.\
                    find_all("div", {"class": "comment"})
                node_attr['com_children'] = [
                    child.get("id") for child in child_comments if
                    depth_search in child["class"]]
            else:
                node_attr['com_children'] = []
        except (AttributeError, TypeError):
            node_attr['com_children'] = []
        # adding thread_url
        node_attr['com_thread'] = self.data.thread_url
        self.create_node(com_id, node_attr)


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
