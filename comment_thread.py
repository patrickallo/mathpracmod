"""
Module that includes the comment_thread parent class, and subclass for Polymath.
It uses: requests, BeautifullSoup, and networkx.DiGraph.
"""

import sys
import requests
from datetime import datetime
import yaml

from bs4 import BeautifulSoup
import networkx as nx
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import access_classes as ac
import export_classes as ec
#import matplotlib.dates as mdates

def main(urls, thread_type="Polymath"):
    """Created thread based on supplied url, and draws graph."""
    if thread_type == "Polymath":
        the_threads = [CommentThreadPolymath(url) for url in urls]
        an_mthread = MultiCommentThread(*the_threads)
        #a_select = a_thread.graph.nodes()[5:15] # does not only select level_1 nodes!
        an_mthread.draw_graph()
        #a_thread.print_nodes(*a_select)
    else:
        print "No other types currently implemented."

class CommentThread(ac.ThreadAccess, object):
    """
    Parent class for storing a comment thread to a WordPress post in a directed graph.
    Subclasses have the appropriate parsing method.

    Methods:
        parse_thread: not implemented.
        from thread_access
        comment_report: returns report on comment as dict
        print_nodes: prints data of requested node out in yaml.
        print_html: prints out html-soup of selected node.

    Attributes:
        req: request from url
        soup: BeautifullSoup parsing of content of request
        comments_and_graph: dict of html of comments, and DiGraph of thread-structure.
        graph: DiGraph based on thread
        node_name: dict with nodes as keys and authors as values
        authors: set with author-names (no repetitions, but including pingbacks)
    """
    def __init__(self, url, comments_only):
        self.req = requests.get(url)
        self.soup = BeautifulSoup(self.req.content, 'html5lib')
        self.post_title = self.soup.find("div", {"class": "post"}).find("h3").text
        self.post_content = self.soup.find("div", {"class":"storycontent"}).find_all("p")
        self.comments_and_graph = self.parse_thread(self.soup)
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
        #print "Empty dict and graph are returned"
        #return {'as_dict': {}, 'as_graph': nx.DiGraph()}
        raise NotImplementedError("Subclasses should implement this!")

    ## Accessor methods
    


class MultiCommentThread(ac.ThreadAccess, ec.GraphExport, object):
    """
    Combines graphs of multiple comment_threads for uniform author colouring.
    Main uses: drawing graphs, and supply to author_network.
    Drawing of separate threads should also use this class.

    Attributes:
        threads_graph: nx.DiGraph
        author_color: dict with authors as keys and colors (ints) as values
        node_name: dict with nodes as keys and authors as values

    Methods:
        add_thread: mutator method for adding thread to multithread.
                    This method is called by init.
        from thread_access
        print_nodes: prints data of requested node out in yaml.
        print_html: prints out html-soup of selected node.
        comment_report: accessor method that returns structural info about a single comment.
        draw_graph: accessor method that draws (a selection of) the mthread graph.
        """

    def __init__(self, *threads):
        self.threads_graph = nx.DiGraph()
        self.author_color = {}
        self.node_name = {}
        for thread in threads:
            self.add_thread(thread)

    ## Mutator methods
    def add_thread(self, thread):
        """Adds new (non-overlapping) thread by updating author_color and DiGraph"""
        # do we need to check for non-overlap?
        self.new_authors = thread.authors.difference(self.author_color.keys())
        self.new_colors = {a: c for (a, c) in
                           zip(self.new_authors,
                               range(len(self.author_color),
                                     len(self.author_color) + len(self.new_authors)))}
        self.author_color.update(self.new_colors)
        self.node_name.update(thread.node_name)
        self.threads_graph = nx.compose(self.threads_graph, thread.graph)

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
                     for (node_id, data) in self.threads_graph.nodes_iter(data=True)}
        node_color = {node_id: (self.author_color[self.node_name[node_id]])
                      for node_id in self.threads_graph.nodes()}
        # actual drawing
        nx.draw_networkx(self.threads_graph, positions, with_labels=show_labels,
 w                        node_size=20,
                         font_size=8,
                         width=.5,
                         nodelist=node_color.keys(),
                         node_color=node_color.values(),
                         cmap=plt.cm.Accent)
        plt.show()


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath"""
    def __init__(self, url, comments_only=True):
        super(CommentThreadPolymath, self).__init__(url, comments_only)

    @classmethod
    def parse_thread(cls, a_soup):
        """ Creates an nx.DiGraph from the comment_soup, and returns both soup and graph.
        This method is only used by init."""
        a_graph = nx.DiGraph()
        a_dict = {}
        the_comments = a_soup.find("ol", {"id": "commentlist"})
        if the_comments:
            all_comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li")
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
            convert_author = {"gagika" : "Gagik Amirkhanyan"}
            try:
                com_author = comment.find("cite").find("span").text
            except AttributeError as err:
                print err, comment.find("cite")
                com_author = "unable to resolve"
            com_author = convert_author[com_author] if com_author in convert_author else com_author
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
            attr = {
                "com_type" : com_class[0],
                "com_depth" : com_depth,
                "com_content" : com_all_content[:-1],
                "com_timestamp" : time_stamp,
                "com_author" : com_author,
                "com_author_url" : com_author_url,
                "com_children" : child_comments}
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        # creating edges
        for node_id, children in nx.get_node_attributes(a_graph, "com_children").iteritems():
            if children:
                a_graph.add_edges_from(((node_id, child) for child in children))
        return {'as_dict': a_dict, 'as_graph': a_graph}


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except IndexError:
        print "testing with Minipolymath 4"
        main('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')
