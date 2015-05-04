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

def list_comments(a_soup, depth=1, maxdepth=5, structured=True, as_series=True):
    """Takes comments from bs4-resultset and yields a dict for each item. Returning lists instead of Series is for use with json.dumps."""
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
        except AttributeError:
            website = None
        try:
            if depth < maxdepth:
                children = list_comments(comment, depth=depth+1, as_series=as_series)
            else:
                children = Series([]) if as_series else []
        except AttributeError:
            children = Series([]) if as_series else []
        newdict = {
            "com-id": comment.get("id"),
            "content": [item.text for item in comment.find("div", {"class":"comment-author vcard"}).find_all("p")],
            "auth": {"name": comment.find("cite").find("span").text,
            "homepage": website},
            "type": comment.get("class")[0],
            "inside-comments": children}
        output.append(newdict)
    if as_series:
        the_index = range(1, len(output)+1)
        return Series(output, index=the_index)
    else:
        return output


def author_list_and_count(a_soup):
    """ takes bs4-soup, and returns dict with count of authors by first creating flat comment-list"""
    return Counter((comment["auth"]["name"] for comment in list_comments(a_soup, structured=False, as_series=False)))

def graph_comments(series_of_comments, start=None, include_pingback=True): # alternative would be to compose graphs
    """takes series_of_comments and return nx.DiGraph by calling one of the underlying functions depending on start"""
    a_graph = nx.DiGraph()
    def graph_edges(i, series_of_comments):
        """takes index-number and series_of_comments and adds nodes and edges to nx.DiGraph"""
        children = series_of_comments[i]["inside-comments"] if include_pingback else Series([comment for comment in series_of_comments[i]["inside-comments"] if comment['type'] == "comment"])
        if children.empty:
            return
        else:
            a_graph.add_edges_from(((series_of_comments[i]["com-id"], child["com-id"]) for child in children))
            for i in children.index:
                graph_edges(i, children)
    def graph_nodes(series_of_comments):
        """takes series_of_comments: for each comment it adds nodes to nx.DiGraph and calls graph_children"""
        series_of_comments = series_of_comments if include_pingback else Series([comment for comment in series_of_comments if comment['type'] == "comment"])
        for (i, comment) in series_of_comments.iteritems():
            a_graph.add_node(comment["com-id"])
            graph_edges(i, series_of_comments)
    if not start:
        graph_nodes(series_of_comments)
    else:
        graph_edges(start, series_of_comments)
    return a_graph

def add_attributes_graph(series_of_comments, a_graph):
    """adds attributes to graph"""
    for comment in series_of_comments:
        print comment["com-id"]
        a_graph.node[comment["com-id"]]["author"] = comment["auth"]["name"]
        a_graph.node[comment["com-id"]]["content"] = comment["content"]
    return a_graph

###########################
## assignments and calls ##
###########################

# this should move to functions and include error-handling
URL = "http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/"
REQ = requests.get(URL)
SOUP = BeautifulSoup(REQ.content)

# for (key, value) in author_list_and_count(SOUP).iteritems():
#     print key, value

STRUCTURED = list_comments(SOUP)
STRUCTURED2 = list_comments(SOUP, as_series=False)
PLAIN = list_comments(SOUP, structured=False, as_series=False)

GRAPH = graph_comments(STRUCTURED, include_pingback=True)
DECORATED_GRAPH = add_attributes_graph(PLAIN, GRAPH)
print json.dumps(DECORATED_GRAPH.nodes(data=True), sort_keys=True, indent=4)

#print "inside comments is of type: ", type(structured.iget(5)["inside-comments"])

#nx.draw(GRAPH, with_labels=True, arrows=True)
for node in GRAPH:
    nx.draw(node)
plt.show()
