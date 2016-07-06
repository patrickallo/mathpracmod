
"""
Module that includes the CommentThread parent class,
subclasses for Polymath, Gil Kalai, Gowers, SBSeminar and Terry Tao,
and the MultiCommentThread.
Main libraries used: BeautifullSoup and networkx.
"""

# Imports
import argparse
from collections import OrderedDict
from concurrent import futures
import datetime
import logging
from operator import methodcaller
import re
import sys
from urllib.parse import urlparse
import yaml

from bs4 import BeautifulSoup
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import requests
from sklearn.cluster import MeanShift, estimate_bandwidth


import access_classes as ac
import multi_comment_thread as mc
import text_functions as tf

# Loading settings
SETTINGS, CMAP = ac.load_settings()
CONVERT, LASTS, *_ = ac.load_yaml("author_convert.yaml", "lasts.yaml")

try:
    with open("lasts.yaml", "r") as lasts_file:
        LASTS = yaml.safe_load(lasts_file.read())
except IOError as err:
    logging.warning("Could not open last_comments: %s", err)
    LASTS = {}


# Pre-declaring dict for selection of subclass of CommentThread
THREAD_TYPES = {}
# actions to be used as argument for --more
ACTIONS = {"graph": "draw_graph",
           "growth": "plot_growth",
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
    with open("DATA/" + project.replace(" ", "") + ".csv", "r") as data_input:
        data = pd.read_csv(data_input, index_col="Ord")
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
            assert list(title_thread.keys()) == data['title'].tolist()
        except AssertionError:
            logging.warning("Threads not in proper order")
        except TypeError:
            logging.warning("Casting to list or comparison failed")
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
        self.is_research = is_research
        self.url = url
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
                logging.exception("Could not connect: %s", err)
                sys.exit(1)
            joblib.dump(self._req, reqfile)
        # faster parsers do not work
        self.soup = BeautifulSoup(self._req.content, SETTINGS['parser'])
        self.parse_thread()
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

    def parse_thread(self):
        """Abstract method: raises NotImplementedError."""
        raise NotImplementedError("Subclasses should implement this!")

    @staticmethod
    def get_seq_nr(content, url):
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

    def create_node(self, *args):
        """adds node for com_id and attributes from *args to self.graph"""
        try:
            assert len(args) == 11
        except AssertionError:
            logging.warning(
                "Wrong number of arguments. Expected 11, received %i",
                len(args))
            print(args)
        com_id = args[1]
        attr = {
            "com_type": args[2][0],
            "com_depth": args[3],
            "com_content": " ".join(args[4]),
            "com_timestamp": args[5],
            "com_author": args[6],
            "com_author_url": args[7],
            "com_children": args[8],
            "com_thread": args[9],
            "com_seq_nr": args[10]}
        attr['com_tokens'], attr['com_stems'] = tf.tokenize_and_stem(
            attr['com_content'])
        self.graph.add_node(com_id)
        logging.debug("Created %s", com_id)
        for (key, value) in attr.items():
            self.graph.node[com_id][key] = value

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
        return com_author

    @staticmethod
    def parse_timestamp(time_stamp, date_format):
        """Parses time_stamp to datetime object."""
        try:
            time_stamp = datetime.datetime.strptime(
                time_stamp, date_format)
        except ValueError as err:
            logging.warning("%s: datetime failed", err)
        return time_stamp

    def remove_comments(self):
        """Lookups last to be included comments in LASTS and
        removes all later comments."""
        try:
            last_comment = LASTS[self.url]
        except KeyError:
            logging.warning("Moving on without removing comments for %s",
                            self.url)
            return
        try:  # TODO: this fails for comment-24559 in post4 of pm10
            last_date = self.graph.node[last_comment]["com_timestamp"]
        except KeyError:
            logging.warning("Node %s not in network", last_comment)
            return
        logging.debug(
            "Removing from %s, %s in %s",
            last_comment, last_date, self.url)
        to_remove = [node for (node, date) in nx.get_node_attributes(
            self.graph, "com_timestamp").items() if date > last_date]
        if to_remove:
            logging.debug("Removing comments %s", to_remove)
            self.graph.remove_nodes_from(to_remove)

    def cluster_comments(self):
        """
        Clusters comments based on their timestamps and
        assigns cluster-membership as attribute to nodes.
        """
        the_nodes = self.graph.nodes()
        if len(the_nodes) < 7:
            logging.warning(
                "Skipped clustering for %s, only %i comments",
                self.post_title,
                len(the_nodes))
            for node in the_nodes:
                self.graph.node[node]['cluster_id'] = 1
        else:
            com_ids, stamps = zip(
                *((node, data["com_timestamp"])
                  for node, data in self.graph.nodes_iter(data=True)))
            data = DataFrame(
                {'timestamps': stamps}, index=com_ids).sort_values(
                    by='timestamps')
            epoch = data.ix[0, 'timestamps']
            data['timestamps'] = data['timestamps'].apply(
                lambda timestamp: (timestamp - epoch).total_seconds())
            cluster_data = data.as_matrix()
            # TODO: need more refined way to set quantile
            try:
                bandwidth = estimate_bandwidth(cluster_data, quantile=0.05)
                logging.info("Bandwidth estimated at %f", bandwidth)
                mshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                mshift.fit(cluster_data)
            except ValueError:
                logging.info("Setting bandwidth with .3 quantile")
                try:
                    bandwidth = estimate_bandwidth(cluster_data)
                    logging.info("Bandwidth estimated at %f", bandwidth)
                    mshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                except ValueError:
                    bandwidth = estimate_bandwidth(cluster_data, quantile=1)
                    bandwidth = bandwidth if bandwidth > 0 else .5
                    logging.info("Bandwidth estimates at %f", bandwidth)
                    mshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                try:
                    mshift.fit(cluster_data)
                except ValueError as err:
                    logging.warning(
                        "Could not cluster %s: %s",
                        self.post_title, err)
                    sys.exit(1)
            labels = mshift.labels_
            unique_labels = np.sort(np.unique(labels))
            logging.info("Found %i clusters in %s",
                         len(unique_labels), self.post_title)
            try:
                assert len(labels) == len(cluster_data)
                # assert (unique_labels == np.arange(len(unique_labels))).all()
            except AssertionError as err:
                logging.warning("Mismatch cluster-labels: %s", err)
                print(unique_labels)
                print(labels)
            data['cluster_id'] = labels
            for com_id in data.index:
                self.graph.node[com_id]['cluster_id'] = data.ix[
                    com_id, 'cluster_id']


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath Blog, with method for actual parsing. """
    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadPolymath, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post"}).find("h3").text.strip()
        self.post_content = self.soup.find(
            "div", {"class": "storycontent"}).find_all("p")
        self.cluster_comments()

    def parse_thread(self):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        self.graph = nx.DiGraph()
        the_comments = self.soup.find("ol", {"id": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            self.process_comment(comment)
        self.create_edges()
        self.remove_comments()

    def process_comment(self, comment):
        """Processes soup from single comment, and creates node with
        corresponding attributes."""
        # identify id, class, depth and content
        com_id = comment.get("id")
        com_class = comment.get("class")
        com_depth = next(int(word[6:]) for word
                         in com_class if word.startswith("depth-"))
        com_all_content = [item.text for item in
                           comment.find(
                               "div",
                               {"class": "comment-author vcard"}).find_all(
                                   "p")]
        # getting and converting author_name
        com_author = self.get_conv_author(
            comment,
            lambda comment: comment.find("cite").find("span").text)
        # creating timeStamp (and time is poped from all_content)
        time_stamp = " ".join(com_all_content.pop().split()[-7:])[2:]
        time_stamp = self.parse_timestamp(time_stamp, "%B %d, %Y @ %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            com_author_url = comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          com_author)
            com_author_url = None
        # get sequence-number of comment (if available)
        seq_nr = self.get_seq_nr(com_all_content, self.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(com_depth + 1)
            child_comments = comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            child_comments = [child.get("id") for child in child_comments]
        except AttributeError:
            child_comments = []
        self.create_node(self, com_id,
                         com_class,
                         com_depth,
                         com_all_content,
                         time_stamp,
                         com_author,
                         com_author_url,
                         child_comments,
                         self.thread_url,
                         seq_nr)


class CommentThreadGilkalai(CommentThread):
    """ Child class for Gil Kalai Blog, with method for actual parsing. """
    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadGilkalai, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.soup.find(
            "div", {"id": "content"}).find(
                "h2", {"class": "entry-title"}).text.strip()
        self.post_content = self.soup.find(
            "div", {"class": "entry-content"}).find_all("p")
        self.cluster_comments()

    def parse_thread(self):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        self.graph = nx.DiGraph()
        the_comments = self.soup.find("ol", {"class": "commentlist"})
        if the_comments:
            # NOTE: Pingbacks have no id and are ignored
            all_comments = the_comments.find_all("li", {"class": "comment"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            self.process_comment(comment)
        self.create_edges()
        self.remove_comments()

    def process_comment(self, comment):
        """Processes soup from single comment, and creates node with
        corresponding attributes."""
        # identify id, class, depth and content
        com_id = comment.find("div").get("id")
        com_class = comment.get("class")
        com_depth = next(int(word[6:]) for word in com_class if
                         word.startswith("depth-"))
        com_all_content = [item.text for item in comment.find(
            "div", {"class": "comment-body"}).find_all("p")]
        # getting and converting author_name
        com_author = self.get_conv_author(
            comment,
            lambda comment: comment.find("cite").text.strip())
        # creating timeStamp
        time_stamp = comment.find(
            "div", {"class": "comment-meta commentmetadata"}).text.strip()
        time_stamp = self.parse_timestamp(time_stamp,
                                          "%B %d, %Y at %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            com_author_url = comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          com_author)
            com_author_url = None
        # get sequence-number of comment (if available)
        seq_nr = self.get_seq_nr(com_all_content, self.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(com_depth + 1)
            child_comments = comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            child_comments = [child.find("div").get("id") for
                              child in child_comments]
        except AttributeError:
            child_comments = []
        self.create_node(self, com_id,
                         com_class,
                         com_depth,
                         com_all_content,
                         time_stamp,
                         com_author,
                         com_author_url,
                         child_comments,
                         self.thread_url,
                         seq_nr)


class CommentThreadGowers(CommentThread):
    """ Child class for Gowers Blog, with method for actual parsing."""
    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadGowers, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post"}).find("h2").text.strip()
        self.post_content = self.soup.find(
            "div", {"class": "post"}).find(
                "div", {"class": "entry"}).find_all("p")
        self.cluster_comments()

    def parse_thread(self):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        self.graph = nx.DiGraph()
        the_comments = self.soup.find("ol", {"class": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            self.process_comment(comment)
        self.create_edges()
        self.remove_comments()

    def process_comment(self, comment):
        """Processes soup from single comment, and creates node with
        corresponding attributes."""
        # identify id, class, depth and content
        com_id = comment.get("id")
        com_class = comment.get("class")
        com_depth = next(int(word[6:]) for
                         word in com_class if word.startswith("depth-"))
        com_all_content = [item.text for item in
                           comment.find_all("p")]
        # getting and converting author_name
        com_author = self.get_conv_author(
            comment,
            lambda comment: comment.find("cite").text.strip())
        # creating timeStamp
        time_stamp = comment.find("small").find("a").text
        time_stamp = self.parse_timestamp(time_stamp,
                                          "%B %d, %Y at %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            com_author_url = comment.find("cite").find(
                "a", {"rel": "external nofollow"}).get("href")
        except AttributeError:
            logging.debug("Could not resolve author_url for %s",
                          com_author)
            com_author_url = None
        # get sequence-number of comment (if available)
        seq_nr = self.get_seq_nr(com_all_content, self.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(com_depth + 1)
            child_comments = comment.find(
                "ul", {"class": "children"}).find_all(
                    "li", {"class": depth_search})
            child_comments = [child.get("id") for child in child_comments]
        except AttributeError:
            child_comments = []
        self.create_node(self, com_id,
                         com_class,
                         com_depth,
                         com_all_content,
                         time_stamp,
                         com_author,
                         com_author_url,
                         child_comments,
                         self.thread_url,
                         seq_nr)


class CommentThreadSBSeminar(CommentThread):
    """
    Child class for Secret Blogging Seminar, with method for actual parsing.
    """
    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadSBSeminar, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.soup.find(
            "article").find("h1", {"class": "entry-title"}).text.strip()
        self.post_content = self.soup.find(
            "div", {"class": "entry-content"}).find_all("p")
        self.cluster_comments()

    def parse_thread(self):
        """
        Creates and returns an nx_DiGraph from the comment_soup.
        This method is only used by init.
        """
        self.graph = nx.DiGraph()
        the_comments = self.soup.find("ol", {"class": "comment-list"})
        if the_comments:
            all_comments = the_comments.find_all("li", {"class": "comment"})
        else:
            all_comments = []
        for comment in all_comments:
            self.process_comment(comment)
        self.create_edges()
        self.remove_comments()

    def process_comment(self, comment):
        """Processes soup from single comment, and creates node with
        corresponding attributes."""
        # identify id, class, depth and content
        com_id = comment.get("id")
        com_class = comment.get("class")
        com_depth = next(int(word[6:]) for word in com_class if
                         word.startswith("depth-"))
        com_all_content = [item.text for item in
                           comment.find(
                               "div",
                               {"class": "comment-content"}).find_all("p")]
        # getting and converting author_name and getting url
        com_author_and_url = comment.find(
            "div", {"class": "comment-author"}).find(
                "cite", {"class": "fn"})
        try:
            com_author = com_author_and_url.find("a").text
            com_author_url = com_author_and_url.find("a").get("href")
        except AttributeError:
            try:
                com_author = com_author_and_url.text
                com_author_url = None
            except AttributeError:
                logging.debug("Could not resolve author_url for %s",
                              com_author)
                com_author = "unable to resolve"
        com_author = CONVERT[com_author] if\
            com_author in CONVERT else com_author
        # creating timeStamp
        time_stamp = comment.find(
            "div", {"class": "comment-metadata"}).find("time").get(
                "datetime")
        time_stamp = self.parse_timestamp(time_stamp,
                                          "%Y-%m-%dT%H:%M:%S+00:00")
        # get sequence-number of comment (if available)
        seq_nr = self.get_seq_nr(com_all_content, self.thread_url)
        # make list of child-comments (only id's) VOID IN THIS CASE
        child_comments = []
        self.create_node(self, com_id,
                         com_class,
                         com_depth,
                         com_all_content,
                         time_stamp,
                         com_author,
                         com_author_url,
                         child_comments,
                         self.thread_url,
                         seq_nr)


class CommentThreadTerrytao(CommentThread):
    """ Child class for Tao Blog, with method for actual parsing."""
    def __init__(self, url, is_research, comments_only=True):
        super(CommentThreadTerrytao, self).__init__(
            url, is_research, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post-meta"}).find("h1").text.strip()
        self.post_content = self.soup.find(
            "div", {"class": "post-content"}).find_all("p")
        self.cluster_comments()

    def parse_thread(self):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        self.graph = nx.DiGraph()
        the_comments = self.soup.find("div", {"class": "commentlist"})
        if the_comments:
            # this seems to ignore the pingbacks
            all_comments = the_comments.find_all("div", {"class": "comment"})
            all_comments += the_comments.find_all("div", {"class": "pingback"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            self.process_comment(comment)
        self.create_edges()
        self.remove_comments()

    def process_comment(self, comment):
        """Processes soup from single comment, and creates node with
        corresponding attributes."""
        # identify id, class, depth and content
        com_id = comment.get("id")
        com_class = comment.get("class")
        com_depth = next(int(word[6:]) for word in com_class if
                         word.startswith("depth-"))
        com_all_content = [item.text for item in
                           comment.find_all("p")][2:]
        # getting and converting author_name
        com_author = self.get_conv_author(
            comment,
            lambda comment: comment.find(
                "p", {"class": "comment-author"}).text)
        # creating timeStamp
        time_stamp = comment.find(
            "p", {"class": "comment-permalink"}).find("a").text
        time_stamp = self.parse_timestamp(time_stamp,
                                          "%d %B, %Y at %I:%M %p")
        # getting href to comment author webpage (if available)
        try:
            com_author_url = comment.find(
                "p", {"class": "comment-author"}).find("a").get("href")
        except AttributeError:
            logging.debug(
                "Could not resolve author_url for %s", com_author)
            com_author_url = None
        # get sequence-number of comment (if available)
        seq_nr = self.get_seq_nr(com_all_content, self.thread_url)
        # make list of child-comments (only id's)
        try:
            depth_search = "depth-" + str(com_depth + 1)
            if comment.next_sibling.next_sibling['class'] == ['children']:
                child_comments = comment.next_sibling.next_sibling.\
                    find_all("div", {"class": "comment"})
                child_comments = [child.get("id") for child in
                                  child_comments if depth_search in
                                  child["class"]]
            else:
                child_comments = []
        except (AttributeError, TypeError):
            child_comments = []
        self.create_node(self, com_id,
                         com_class,
                         com_depth,
                         com_all_content,
                         time_stamp,
                         com_author,
                         com_author_url,
                         child_comments,
                         self.thread_url,
                         seq_nr)

THREAD_TYPES = {"Polymathprojects": CommentThreadPolymath,
                "Gilkalai": CommentThreadGilkalai,
                "Gowers": CommentThreadGowers,
                "Sbseminar": CommentThreadSBSeminar,
                "Terrytao": CommentThreadTerrytao}


if __name__ == '__main__':
    PARSER = ac.make_arg_parser(
        ACTIONS.keys, SETTINGS['project'],
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
