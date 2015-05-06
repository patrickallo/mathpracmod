"""
Module that includes the comment_thread class.
It uses: requests, BeautifullSoup, and networkx.DiGraph.
"""

import sys
import requests
from bs4 import BeautifulSoup
import networkx as nx
from datetime import datetime
from collections import Counter
import yaml
import matplotlib.pyplot as plt
import numpy as np

def main(url):
    a_thread = CommentThread(url)

class CommentThread(object):
    """
    Object that stores a comment thread to a WordPress post in a directed graph.

    Methods:
        parse_thread: uses soup to create dict of html of comments,
        and graph with decorated nodes and edges.
        print_nodes: prints data of requested node out in yaml.
        print_html: prints out html-soup of selected node.

    Attributes:
        url: the url of the blog-post
        req: request from url
        soup: BeautifullSoup parsing of content of request
        comments_and_graph: dict of html of comments, and DiGraph of thread-structure.
        graph: DiGraph based on thread
    """
    def __init__(self, url):
        self.url = url
        self.req = requests.get(url)
        self.soup = BeautifulSoup(self.req.content)
        self.comments_and_graph = self.parse_thread(self.soup)
        self.graph = self.comments_and_graph["as_graph"]

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
            com_depth = next(int(word[-1]) for word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find("div", {"class":"comment-author vcard"}).find_all("p")]
            ## add complete html to dict
            a_dict[com_id] = comment
            # creating timeStamp (currently as string, but should become date-object)
            time_stamp = " ".join(com_all_content[-1].split()[-7:])[2:]
            try:
                time_stamp = datetime.strptime(time_stamp, "%B %d, %Y @ %I:%M %p")
            except AttributeError as err:
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
                child_comments = comment.find("ul", {"class": "children"}).find_all("li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties to be used as attributes of comment nodes
            attr = {
                "com_type" : com_class[0],
                "com_depth" : com_depth,
                "com_content" : com_all_content[:-1],
                "com_timestamp" : str(time_stamp), # provisionally string-conversion for json dumps
                "com_author" : comment.find("cite").find("span").text,
                "com_author_url" : com_author_url,
                "com_children" : child_comments}
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        # creating edges
        for (node_id, data) in a_graph.nodes_iter(data=True):
            if data["com_children"]:
                a_graph.add_edges_from(((node_id, child) for child in data["com_children"]))
        return {'as_dict': a_dict, 'as_graph': a_graph}

    def get_post(self):
        """Returns title and body of blog-post"""
        story_title = self.soup.find("div", {"class": "post"}).find("h3").text
        story_content = self.soup.find("div", {"class":"storycontent"}).find_all("p")
        return (story_title, story_content)

    def author_count(self):
        """returns dict with count of authors"""
        return Counter((data["com_author"] for (node_id, data) in self.graph.nodes_iter(data=True)))
        
    def plot_author_count(self):
        """shows plot of author_count"""
        labels, values = zip(*self.author_count().items())
        indexes = np.arange(len(labels))
        plt.bar(indexes, values, 1)
        plt.xticks(indexes + 0.5, labels, rotation='vertical')
        plt.show()
    
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

    def draw_graph(self):
        """Draws and shows graph."""
        nx.draw(self.graph, with_labels=True, arrows=True)
        plt.show()

if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError:
        print "testing with Minipolymath 4"
        main('http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/')

