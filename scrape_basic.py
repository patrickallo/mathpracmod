""" basic version to be used with single posts from WordPress"""

import requests
from bs4 import BeautifulSoup
from pandas import Series

url = "http://polymathprojects.org/2012/07/12/minipolymath4-project-imo-2012-q3/"
req = requests.get(url)
soup = BeautifulSoup(req.content)


def list_comments(a_soup, depth=1):
    """takes comments from bs4-resultset and yields a dict for each item"""
    output = []
    depth_search = "depth-" + str(depth)
    if depth == 1:
        comments = a_soup.find("ol", {"id": "commentlist"}).find_all("li", {"class": depth_search})
    else:
        comments = a_soup.find("ul", {"class": "children"}).find_all("li", {"class": depth_search})
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
                children = None
        except:
            children = None
        newdict = { "com-id": comment.get("id"),
                "text": [item.text for item in comment.find_all("p")],
                "auth": {"name": comment.find("cite").find("span").text, "homepage": website},
                "inside-comments": children}
        output.append(newdict)
    return output

def to_series(lst):
    """returns Series with 1-based index"""
    return Series(lst, index=range(1, len(lst)+1))


print to_series(list_comments(soup))
