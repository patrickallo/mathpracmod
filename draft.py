""" basic version to be used with single posts from WordPress"""

import requests
from bs4 import BeautifulSoup
from pandas import Series
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
#import pygraphviz # not available

##########################
## function definitions ##
##########################

def list_comments(a_soup, depth=1, maxdepth=3, structured=True, as_series=True):
    """takes comments from bs4-resultset and yields a dict for each item"""
    output = []
    depth_search = "depth-" + str(depth)
    if structured:
        if depth == 1:
            comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li", {"class": depth_search})
        else:
            comments = a_soup.find("ul", {"class": "children"}).find_all("li", {"class": depth_search})
    else:
        comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li")
    for comment in comments:
        try:
             website = comment.find("cite").find("a", {"rel": "external nofollow"}).get("href")
        except:
             website  = None
        try:
            if depth < maxdepth:
                children = list_comments(comment, depth=depth+1, as_series=as_series)
            else:
                children = Series([]) if as_series else []
        except:
            children = Series([]) if as_series else []
        newdict = { "com-id": comment.get("id"),
                "content": [item.text for item in comment.find("div", {"class": "comment-author vcard"}).find_all("p")],
                "auth": {"name": comment.find("cite").find("span").text, "homepage": website},
                "inside-comments": children}
        output.append(newdict)
    if as_series:
        the_index = range(1, len(output)+1)
        return Series(output, index=the_index)
    else:
        return output

def navigate_comments(series_of_comments, do_all_child_notes, do_all_links, do_node, output=[], start=None):
    def do_children(i, series_of_comments):
        children = series_of_comments[i]["inside-comments"]
        if children.empty:
            return
        else:
            do_all_child_notes(i, children) # acts on nodes
            do_all_links(i, children)
            for i in children.index:
                do_children(i, children)
    def do_all(series_of_comments):
        for (i, comment) in series_of_comments.iteritems():
            do_node()
            do_children(i, series_of_comments)
    if not start:
        do_all(series_of_comments)
    else:
        do_children(start, series_of_comments)
    print type(output)
    nx.draw(output,with_labels=True,arrows=True)
    plt.show()
    
def graph_comments(series_of_comments, start=None):
    graph = nx.DiGraph()
    def graph_children(i, children):
        graph.add_nodes_from((child["com-id"] for child in children))
        graph.add_edges_from(((series_of_comments[i]["com-id"], child["com-id"]) for child in children))
    def graph_edges(i, children):
        graph.add_edges_from(((series_of_comments[i]["com-id"], child["com-id"]) for child in children))
    def graph_node():
        graph.add_node(comment["com-id"])
    navigate_comments(series_of_comments, graph_children, graph_edges, graph_node, output=graph, start=start)

url = "http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/"
req = requests.get(url)
soup = BeautifulSoup(req.content)


structured = list_comments(soup)
graph_comments(structured, start=11) # does not work correctly 
