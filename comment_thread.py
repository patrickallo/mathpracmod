"""
Module that includes the comment_thread class.
It uses: requests, BeautifullSoup, and networkx.DiGraph.
"""

import requests
from bs4 import BeautifulSoup
import networkx as nx
from datetime import datetime
import json
import matplotlib.pyplot as plt

class CommentThread(object):
    """
    Object that stores a comment thread to a WordPress post in a directed graph.

    Methods:
        to be included

    Attributes:
        url: the url of the blog-post
        req: request from url
        soup: BeautifullSoup parsing of content of request
        graph: DiGraph based on thread
    """
    def __init__(self, url):
        self.url = url
        self.req = requests.get(url)
        self.soup = BeautifulSoup(self.req.content)
        self.graph = self.create_graph(self.soup)

    def create_graph(self, a_soup):
        """ Creates an nx.DiGraph from the comment_soup. This method is only used by init."""
        a_graph = nx.DiGraph()
        all_comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li")
        for comment in all_comments:
            # identify id and class
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[-1]) for word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in comment.find("div", {"class":"comment-author vcard"}).find_all("p")]
            # creating timeStamp (currently as string, but should become date-object)
            time_stamp = " ".join(com_all_content[-1].split()[-7:])[2:]
            try:
                time_stamp = datetime.strptime(time_stamp, "%B %d, %Y @ %I:%M %p")
            except AttributeError:
                print "datetime failed, replaced by: ", time_stamp
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find("a", {"rel": "external nofollow"}).get("href")
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
        return a_graph

A = CommentThread("http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/")
print json.dumps(A.graph.node['comment-7579'], sort_keys=True, indent=4)

nx.draw(A.graph, with_labels=True, arrows=True)
plt.show()
