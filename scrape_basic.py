""" basic version to be used with single posts from WordPress"""

import requests
from bs4 import BeautifulSoup
from pandas import Series
import json
import networkx as nx
import matplotlib.pyplot as plt
#import pygraphviz # not available

url = "http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/"
req = requests.get(url)
soup = BeautifulSoup(req.content)


def list_comments(a_soup, depth=1, structured=True):
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
            if depth < 3:
                newdepth = depth + 1
                children = list_comments(comment, depth=newdepth)
            else:
                children = []
        except:
            children = []
        newdict = { "com-id": comment.get("id"),
                "content": [item.text for item in comment.find("div", {"class": "comment-author vcard"}).find_all("p")],
                "auth": {"name": comment.find("cite").find("span").text, "homepage": website},
                "inside-comments": children}
        output.append(newdict)
    return output

def to_series(lst):
    """returns Series with 1-based index"""
    return Series(lst, index=range(1, len(lst)+1))
    

structured = to_series(list_comments(soup))
plain = list_comments(soup, structured=False)

def graph_comments(a_series):
    a_graph = nx.DiGraph
    top_notes = a_series.index
    a_graph.add_nodes_from(top_notes)
    return a_graph
    
#G = graph_comments(structured)

print json.dumps(structured.iget(4), sort_keys=True, indent=4)

# creating graph
#G = nx.DiGraph()

# this creates all nodes with com-id as label
#for comment in plain:
#    G.add_node(comment["com-id"])

# this only links level-1 comments to child-notes of level-2
#for comment in structured:
#    if comment["inside-comments"]:
#        for child in comment["inside-comments"]:
#            G.add_edge(comment["com-id"], child["com-id"])
#    else:
#        pass
        
#nx.write_dot(G,'test.dot')  # doesn't work yet (related to import pygraphviz)
#pos=nx.graphviz_layout(G,prog='dot')
#nx.draw(G,pos,with_labels=False,arrows=False)
#plt.show()
