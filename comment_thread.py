"""
Module that includes the comment_thread parent class,
and subclasses for Polymath, Gowers and Terry Tao.
Main libraries used: BeautifullSoup and networkx.
"""

# Imports
from datetime import datetime
import requests
import sys
import yaml

from bs4 import BeautifulSoup
from matplotlib.dates import date2num, DateFormatter, DayLocator
import matplotlib.pyplot as plt
import networkx as nx

import access_classes as ac
import export_classes as ec

# Loading settings
with open("settings.yaml", "r") as settings_file:
    SETTINGS = yaml.safe_load(settings_file.read())

with open("author_convert.yaml", "r") as convert_file:
    CONVERT = yaml.safe_load(convert_file.read())

# Main
def main(urls, thread_type="Polymath"):
    """Created thread based on supplied url, and draws graph."""
    # TODO: fix mismatch between many urls and one type
    try:
        the_threads = eval("[CommentThread{}(url) for url in {}]".format(thread_type, urls))
    except ValueError as err:
        print err
        the_threads = []
    an_mthread = MultiCommentThread(*the_threads)
    an_mthread.draw_graph()

# Classes
class CommentThread(ac.ThreadAccessMixin, object):
    """
    Parent class for storing a comment thread to a WordPress post in a directed graph.
    Subclasses have the appropriate parsing method.
    Inherits methods from parent Mixin.

    Attributes:
        req: request from url.
        soup: BeautifullSoup parsing of content of request.
        comments_and_graph: dict of html of comments, and DiGraph of thread-structure.
        post_title: title of post (only supplied by sub-classes).
        post_content: content of post (only supplied by sub-classes).
        graph: DiGraph based on thread (overruled from ThreadAccessMixin).
        node_name: dict with nodes as keys and authors as values (overruled from ThreadAccessMixin).
        authors: set with author-names (no repetitions, but including pingbacks).

    Methods:
        parse_thread: not implemented.
        print_html: prints out html-soup of selected node(s).
        from ThreadAccessMixin:
            comment_report: takes node-id(s), and returns dict with report about node
            print_nodes: takes nodes-id(s), and prints out node-data as yaml. No output.
    """

    def __init__(self, url, comments_only):
        super(CommentThread, self).__init__()
        self.req = requests.get(url)
        self.soup = BeautifulSoup(self.req.content, SETTINGS['parser'])
        self.comments_and_graph = self.parse_thread(self.soup)
        self.post_title = ""
        self.post_content = ""
        ## creates sub_graph and node:author dict based on comments_only
        if comments_only:
            # create node:name dict for nodes that are comments
            self.node_name = {node_id: data['com_author'] for (node_id, data) in
                              self.comments_and_graph["as_graph"].nodes_iter(data=True) if
                              data['com_type'] == 'comment'}
            # create sub_graph based on keys of node_name
            self.graph = self.comments_and_graph["as_graph"].subgraph(self.node_name.keys())
        else:
            self.graph = self.comments_and_graph["as_graph"]
            self.node_name = nx.get_node_attributes(self.graph, 'com_author')
        self.authors = set(self.node_name.values())

    @classmethod
    def parse_thread(cls, a_soup):
        """Abstract method: raises NotImplementedError."""
        raise NotImplementedError("Subclasses should implement this!")

    @classmethod
    def store_attributes(cls,
                         com_class, com_depth, com_all_content, time_stamp,
                         com_author, com_author_url, child_comments):
        """Returns arguments as dict"""
        return {"com_type" : com_class[0],
                "com_depth" : com_depth,
                "com_content" : com_all_content[:-1],
                "com_timestamp" : time_stamp,
                "com_author" : com_author,
                "com_author_url" : com_author_url,
                "com_children" : child_comments}

    @classmethod
    def create_edges(cls, a_graph):
        """Takes nx.DiGraph, adds edges to child_comments and returns nx.DiGraph."""
        for node_id, children in nx.get_node_attributes(a_graph, "com_children").iteritems():
            if children:
                a_graph.add_edges_from(((node_id, child) for child in children))
        return a_graph

    ## Accessor methods
    def print_html(self, *select):
        """Prints out html for selected comments.
        Main use is for troubleshooting. May be deprecated."""
        if select:
            select = self.comments_and_graph["as_dict"].keys() if \
            select[0].lower() == "all" else select
            for key in select:
                try:
                    print self.comments_and_graph["as_dict"][key]
                except KeyError as err:
                    print err, "not found"
        else:
            print "No comment was selected"


class MultiCommentThread(ac.ThreadAccessMixin, ec.GraphExportMixin, object):
    """
    Combines graphs of multiple comment_threads for uniform author colouring.
    Main uses: drawing graphs, and supply to author_network.
    Drawing of separate threads should also use this class.
    Inherits methods from two parent Mixins.

    Attributes:
        graph: nx.DiGraph (overruled from ThreadAccessMixin).
        author_color: dict with authors as keys and colors (ints) as values.
        node_name: dict with nodes as keys and authors as values (overruled from ThreadAccessMixin).

    Methods:
        add_thread: mutator method for adding thread to multithread.
                    This method is called by init.
        draw_graph: accessor method that draws the mthread graph.
        from ThreadAccessMixin:
            comment_report: takes node-id(s), and returns dict with report about node.
            print_nodes: takes nodes-id(s), and prints out node-data as yaml. No output.
        from ThreadExportMixin:
            to_gephi: exports the full graph to gephi.
            to_yaml: exports the full graph to yaml.

    """

    def __init__(self, *threads):
        super(MultiCommentThread, self).__init__()
        self.graph = nx.DiGraph()
        self.author_color = {}
        self.node_name = {}
        for thread in threads:
            self.add_thread(thread)

    ## Mutator methods
    def add_thread(self, thread):
        """Adds new (non-overlapping) thread by updating author_color and DiGraph."""
        # do we need to check for non-overlap?
        self.new_authors = thread.authors.difference(self.author_color.keys())
        self.new_colors = {a: c for (a, c) in
                           zip(self.new_authors,
                               range(len(self.author_color),
                                     len(self.author_color) + len(self.new_authors)))}
        self.author_color.update(self.new_colors)
        self.node_name.update(thread.node_name)
        self.graph = nx.compose(self.graph, thread.graph)

    ## Accessor methods
    def draw_graph(self):
        """Draws and shows graph."""
        show_labels = raw_input("Show labels? (default = no) ")
        show_labels = show_labels.lower() == 'yes'
        # generating colours and positions
        xfact = 1
        yfact = 1 # should be made dependent on timedelta
        positions = {node_id: (data["com_depth"] * xfact,
                               date2num(data["com_timestamp"]) * yfact)
                     for (node_id, data) in self.graph.nodes_iter(data=True)}
        node_color = {node_id: (self.author_color[self.node_name[node_id]])
                      for node_id in self.graph.nodes()}
        # creating axes
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.yaxis.set_major_locator(DayLocator())
        axes.yaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_ticks(range(1, 7))
        # actual drawing
        nx.draw_networkx(self.graph, positions, with_labels=show_labels,
                         node_size=20,
                         font_size=8,
                         width=.5,
                         nodelist=node_color.keys(),
                         node_color=node_color.values(),
                         cmap=plt.cm.Accent,
                         ax=axes)
        plt.show()


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath Blog, with method for actual paring. """
    def __init__(self, url, comments_only=True):
        super(CommentThreadPolymath, self).__init__(url, comments_only)
        self.post_title = self.soup.find("div", {"class": "post"}).find("h3").text
        self.post_content = self.soup.find("div", {"class":"storycontent"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup):
        """ Creates an nx.DiGraph from the comment_soup, and returns both soup and graph.
        This method is only used by init."""
        a_graph = nx.DiGraph()
        a_dict = {}
        the_comments = a_soup.find("ol", {"id": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = [] # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find("div", {"class":"comment-author vcard"}).find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").find("span").text
            except AttributeError as err:
                print err, comment.find("cite")
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if com_author in CONVERT else com_author
            ## add complete html to dict
            a_dict[com_id] = comment
            # creating timeStamp
            time_stamp = " ".join(com_all_content[-1].split()[-7:])[2:]
            try:
                time_stamp = datetime.strptime(time_stamp, "%B %d, %Y @ %I:%M %p")
            except ValueError as err:
                print err, ": datetime failed"
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find("a",
                                                           {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find("ul",
                                              {"class": "children"}).find_all(
                                                  "li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties to be used as attributes of comment nodes
            attr = cls.store_attributes(com_class, com_depth, com_all_content, time_stamp,
                                        com_author, com_author_url, child_comments)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return {'as_dict': a_dict, 'as_graph': a_graph}

class CommentThreadGilkalai(CommentThread):
    """ Child class for Gil Kalai Blog, with method for actual paring. """
    def __init__(self, url, comments_only=True):
        super(CommentThreadGilkalai, self).__init__(url, comments_only)
        self.post_title = self.soup.find("div",
                                         {"id": "content"}).find("h1",
                                                                 {"class": "entry-title"}).text
        self.post_content = self.soup.find("div", {"class":"entry-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup):
        """ Creates an nx.DiGraph from the comment_soup, and returns both soup and graph.
        This method is only used by init."""
        a_graph = nx.DiGraph()
        a_dict = {}
        the_comments = a_soup.find("ol", {"class": "commentlist"})
        if the_comments:
            # NOTE: Pingbacks have no id and are ignored
            all_comments = the_comments.find_all("li", {"class": "comment"})
        else:
            all_comments = [] # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.find("article").get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find("section", {"class":"comment-content"}).find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").text
            except AttributeError as err:
                print err, comment.find("cite")
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if com_author in CONVERT else com_author
            ## add complete html to dict
            a_dict[com_id] = comment
            # creating timeStamp
            time_stamp = comment.find("time").text
            try:
                time_stamp = datetime.strptime(time_stamp, "%B %d, %Y at %I:%M %p")
            except ValueError as err:
                print err, ": datetime failed"
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find("a",
                                                           {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find("ol",
                                              {"class": "children"}).find_all(
                                                  "li", {"class": depth_search})
                child_comments = [child.find("article").get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties to be used as attributes of comment nodes
            attr = cls.store_attributes(com_class, com_depth, com_all_content, time_stamp,
                                        com_author, com_author_url, child_comments)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return {'as_dict': a_dict, 'as_graph': a_graph}

class CommentThreadGowers(CommentThread):
    """ Child class for Gowers Blog, with method for actual paring."""
    def __init__(self, url, comments_only=True):
        super(CommentThreadGowers, self).__init__(url, comments_only)
        self.post_title = self.soup.find("div", {"class": "post"}).find("h2").text
        self.post_content = self.soup.find("div",
                                           {"class": "post"}).find("div",
                                                                   {"class": "entry"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup):
        """ Creates an nx.DiGraph from the comment_soup, and returns both soup and graph.
        This method is only used by init."""
        a_graph = nx.DiGraph()
        a_dict = {}
        the_comments = a_soup.find("ol", {"class": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = [] # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").text
            except AttributeError as err:
                print err, comment.find("cite")
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if com_author in CONVERT else com_author
            ## add complete html to dict
            a_dict[com_id] = comment
            # creating timeStamp
            time_stamp = comment.find("small").find("a").text
            try:
                time_stamp = datetime.strptime(time_stamp, "%B %d, %Y at %I:%M %p")
            except ValueError as err:
                print err, ": datetime failed"
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find("a",
                                                           {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find("ul",
                                              {"class": "children"}).find_all(
                                                  "li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties to be used as attributes of comment nodes
            attr = cls.store_attributes(com_class, com_depth, com_all_content, time_stamp,
                                        com_author, com_author_url, child_comments)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return {'as_dict': a_dict, 'as_graph': a_graph}


class CommentThreadTerrytao(CommentThread):
    """ Child class for Tao Blog, with method for actual paring."""
    def __init__(self, url, comments_only=True):
        super(CommentThreadTerrytao, self).__init__(url, comments_only)
        self.post_title = self.soup.find("div", {"class": "post-meta"}).find("h1").text
        self.post_content = self.soup.find("div", {"class":"post-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup):
        """ Creates an nx.DiGraph from the comment_soup, and returns both soup and graph.
        This method is only used by init."""
        a_graph = nx.DiGraph()
        a_dict = {}
        the_comments = a_soup.find("div", {"class": "commentlist"})
        if the_comments:
            # this seems to ignore the pingbacks
            all_comments = the_comments.find_all("div", {"class": "comment"})
            all_comments += the_comments.find_all("div", {"class": "pingback"})
        else:
            all_comments = [] # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("p", {"class": "comment-author"}).text
            except AttributeError as err:
                print err, comment.find("cite")
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if com_author in CONVERT else com_author
            ## add complete html to dict
            a_dict[com_id] = comment
            # creating timeStamp
            time_stamp = comment.find("p", {"class": "comment-permalink"}).find("a").text
            try:
                time_stamp = datetime.strptime(time_stamp, "%d %B, %Y at %I:%M %p")
            except ValueError as err:
                print err, ": datetime failed"
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("p",
                                              {"class": "comment-author"}).find("a").get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                if comment.next_sibling.next_sibling['class'] == ['children']:
                    child_comments = comment.next_sibling.next_sibling.find_all("div",
                                                                                {"class":"comment"})
                    child_comments = [child.get("id") for child in child_comments
                                      if depth_search in child["class"]]
                else:
                    child_comments = []
            except (AttributeError, TypeError):
                child_comments = []
            # creating dict of comment properties to be used as attributes of comment nodes
            attr = cls.store_attributes(com_class, com_depth, com_all_content, time_stamp,
                                        com_author, com_author_url, child_comments)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return {'as_dict': a_dict, 'as_graph': a_graph}


if __name__ == '__main__':
    ARGUMENTS = sys.argv[1:]
    if ARGUMENTS:
        main(ARGUMENTS)
    else:
        print SETTINGS['msg']
        main([SETTINGS['url']],
             thread_type=SETTINGS['type'])
