"""
Module that includes the comment_thread class.
It uses: requests, BeautifullSoup, and networkx.DiGraph.
"""

import requests
from bs4 import BeautifulSoup
import networkx as nx

class CommentThread():
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

    def create_graph(self, a_soup, depth=1, maxdepth=5): # should become blend of old list_comments, graph_comments and add_attributes
        a_graph = nx.DiGraph()
        depth_search = "depth-" + str(depth)
        if depth == 1:
            comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li", {"class": depth_search})
        else:
            comments = a_soup.find("ul", {"class": "children"}).find_all("li", {"class": depth_search})
        for comment in comments:
            com_id = comment.get("id")
            try:
                com_author_url = comment.find("cite").find("a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            attr = {
                "com_type" : comment.get("class")[0],
                "com_content" : [item.text for item in comment.find("div", {"class":"comment-author vcard"}).find_all("p")],
                "com_author" : comment.find("cite").find("span").text,
                "com_author_url" : com_author_url}
            a_graph.add_node(com_id)
            for (key, value) in attr.iteritems():
                a_graph.node[com_id][key] = value
        return a_graph

a = CommentThread("http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/")