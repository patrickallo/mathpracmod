## this is a script for analysing the totality of Polymath 7 (most recent "long project")

from comment_thread import *
from author_network import *

URLS = {
    1: "http://polymathprojects.org/2012/06/12/polymath7-research-thread-1-the-hot-spots-conjecture/",
    2: "http://polymathprojects.org/2012/06/15/polymath7-research-threads-2-the-hot-spots-conjecture/",
    3: "http://polymathprojects.org/2012/06/24/polymath7-research-threads-3-the-hot-spots-conjecture/",
    4: "http://polymathprojects.org/2012/09/10/polymath7-research-threads-4-the-hot-spots-conjecture/"}

THREADS = {key: CommentThreadPolymath(value, comments_only=False) for (key, value) in URLS.iteritems()}

MTHREAD = MultiCommentThread(*THREADS.values())

ANETW = AuthorNetwork(MTHREAD)

ANETW.draw_graph()