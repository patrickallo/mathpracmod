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

def main(url, thread_type="Polymath"):
    """Created thread based on supplied url, and draws graph."""
    if thread_type == "Polymath":
        an_mthread = MultiCommentThread(CommentThreadPolymath(url))
        #a_select = a_thread.graph.nodes()[5:15] # does not only select level_1 nodes!
        an_mthread.draw_graph("All")
        #a_thread.print_nodes(*a_select)
    else:
        print "No other types currently implemented."

class CommentThread(object):
    """
    Parent class for storing a comment thread to a WordPress post in a directed graph.
    Subclasses have the appropriate parsing method.

    Methods:
        parse_thread: not implemented,
        and graph with decorated nodes and edges.
        print_nodes: prints data of requested node out in yaml.
        print_html: prints out html-soup of selected node.

    Attributes:
        url: the url of the blog-post
        req: request from url
        soup: BeautifullSoup parsing of content of request
        comments_and_graph: dict of html of comments, and DiGraph of thread-structure.
        graph: DiGraph based on thread
        authors: set with author-names (no repetitions, but including pingbacks)
        author_color: dict with author as key and color as value
    """
    def __init__(self, url):
        self.req = requests.get(url)
        self.soup = BeautifulSoup(self.req.content, 'html5lib')
        self.comments_and_graph = self.parse_thread(self.soup)
        self.graph = self.comments_and_graph["as_graph"]
        ## for use by graph, and for author_graph
        self.node_name = nx.get_node_attributes(self.graph, 'com_author')
        self.authors = set(self.node_name.values())
        self.author_color = {a: c for (a, c) in zip(self.authors, range(len(self.authors)))}

    @classmethod
    def parse_thread(cls, a_soup):
        """Abstract method: raises NotImplementedError."""
        #print "Empty dict and graph are returned"
        #return {'as_dict': {}, 'as_graph': nx.DiGraph()}
        raise NotImplementedError("Subclasses should implement this!")

    ## Accessor methods
    def get_post(self):
        """Returns title and body of blog-post"""
        story_title = self.soup.find("div", {"class": "post"}).find("h3").text
        story_content = self.soup.find("div", {"class":"storycontent"}).find_all("p")
        return (story_title, story_content)

    def print_nodes(self, *select):
        """Prints out node-data as yaml. No output."""
        if select:
            select = self.graph.nodes() if select == ("All",) else select
            for comment in select:
                # do something if comment does not exist!
                print "com_id:", comment
                try:
                    print yaml.safe_dump(self.graph.node[comment], default_flow_style=False)
                except KeyError as err:
                    print err, "not found"
                print "---------------------------------------------------------------------"
        else:
            print "No nodes were selected"

    def print_html(self, *select):
        """Prints out html for selected comments."""
        if select:
            select = self.comments_and_graph["as_dict"].keys() if select == ("All",) else select
            for key in select:
                try:
                    print self.comments_and_graph["as_dict"][key]
                except KeyError as err:
                    print err, "not found"
        else:
            print "No comment was selected"


class MultiCommentThread(object):
    """Combines graphs of multiple comment_threads for uniform author colouring.
    Main uses: drawing graphs, and supply to author_network.
    Drawing of separate threads should also use this class."""
    def __init__(self, *threads):
        self.threads_graph = nx.DiGraph()
        self.author_color = {}
        self.node_name = {}
        for thread in threads:
            self.add_thread(thread)

    ## Mutator methods
    def add_thread(self, thread):
        """Adds new (non-overlapping) thread by updating author_color and DiGraph"""
        self.new_authors = thread.authors.difference(self.author_color.keys())
        self.new_colors = {a: c for (a, c) in
                           zip(self.new_authors,
                               range(len(self.author_color),
                                     len(self.author_color) + len(self.new_authors)))}
        self.author_color.update(self.new_colors)
        self.node_name.update(thread.node_name)
        self.threads_graph = nx.compose(self.threads_graph, thread.graph)

    ## Accessor methods
    def comment_report(self, com_id):
        """Takes node-id, and returns dict with report about node."""
        the_node = self.graph.node[com_id]
        the_author = the_node["com_author"]
        descendants = nx.descendants(self.graph, com_id)
        pure_descendants = [i for i in descendants if
                            self.graph.node[i]['com_author'] != the_author]
        direct_descendants = self.graph.out_degree(com_id)
        return {
            "author" : the_author,
            "level of comment" : the_node["com_depth"],
            "direct replies" : direct_descendants,
            "indirect replies (all, pure)" : (len(descendants), len(pure_descendants))
        }

    def draw_graph(self, *select):
        """Draws and shows graph."""
        show_labels = raw_input("Show labels? (default = no) ")
        show_labels = show_labels.lower() == 'yes'
        if select:
            try:
                subtree = self.threads_graph if select[0].lower() == "all" else \
                nx.compose_all(nx.dfs_tree(self.threads_graph, com_id) for com_id in select)
            except AttributeError as err:
                print err, "supply only comment_id's"
            # generating positions
            xfact = 1
            yfact = 1 # should be made dependent on timedelta
            positions = {node_id: (data["com_depth"] * xfact,
                                   date2num(data["com_timestamp"]) * yfact) for
                         (node_id, data) in self.threads_graph.nodes_iter(data=True)
                         if node_id in subtree}
            # attributing colors with dict node: color for subtree
            node_color = {node_id : self.author_color[self.node_name[node_id]]
                          for node_id in subtree}
            # actual drawing
            nx.draw_networkx(subtree, positions, with_labels=show_labels,
                             node_size=20,
                             font_size=8,
                             width=.5,
                             nodelist=node_color.keys(),
                             node_color=node_color.values(),
                             cmap=plt.cm.Accent) # does not work!
            plt.show()
        else:
            print "No comment was selected"


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath"""
    def __init__(self, url):
        super(CommentThreadPolymath, self).__init__(url)

    @classmethod
    def parse_thread(cls, a_soup):
        """ Creates an nx.DiGraph from the comment_soup, and returns both soup and graph.
        This method is only used by init."""
        a_graph = nx.DiGraph()
        a_dict = {}
        all_comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li")
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
            # creating timeStamp (currently as string, but should become date-object)
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
        main(sys.argv[1])
    except IndexError:
        print "testing with Minipolymath 4"
        main('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')

