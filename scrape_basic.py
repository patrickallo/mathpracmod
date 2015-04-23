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

## list_comments does not discriminate between comments and pingbacks

def author_list_and_count(a_soup):
    """ takes bs4-soup, and returns dict with count of authors by first creating flat comment-list"""
    return Counter((comment["auth"]["name"] for comment in list_comments(a_soup, structured=False, as_series=False)))

def graph_comments(series_of_comments, start=None): # alternative would be to compose graphs
    """takes series_of_comments and return nx.DiGraph by calling one of the underlying functions depending on start"""
    G = nx.DiGraph()
    def graph_children(i, series_of_comments):
        """takes index-number and series_of_comments and adds nodes and edges to nx.DiGraph"""
        children = series_of_comments[i]["inside-comments"]
        if children.empty:
            return
        else:
            G.add_nodes_from((child["com-id"] for child in children))
            G.add_edges_from(((series_of_comments[i]["com-id"], child["com-id"]) for child in children))
            for i in children.index:
                graph_children(i, children)
    def graph_all(series_of_comments):
        """takes series_of_comments: for each comment it adds nodes to nx.DiGraph and calls graph_children"""
        for (i, comment) in structured.iteritems():
            G.add_node(comment["com-id"])
            graph_children(i, structured)
    if start == None:
        graph_all(series_of_comments)
    else:
        graph_children(start, series_of_comments)
    print "argument supplied is of type {}".format(type(series_of_type))
    return G

###########################
## assignments and calls ##
###########################

url = "http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/"
req = requests.get(url)
soup = BeautifulSoup(req.content)

for (key, value) in author_list_and_count(soup).iteritems():
    print key, value

# structured = list_comments(soup)
# structured2 = list_comments(soup, as_series=False)
#plain = list_comments(soup, structured=False)

#G = graph_comments(structured, start=11)

#print json.dumps(structured2[10], sort_keys=True, indent=4)
#print "inside comments is of type: ", type(structured.iget(5)["inside-comments"])

#nx.draw(G,with_labels=True,arrows=True)
#plt.show()
