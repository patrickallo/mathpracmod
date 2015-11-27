
"""
Module that includes the comment_thread parent class,
and subclasses for Polymath, Gowers and Terry Tao.
Main libraries used: BeautifullSoup and networkx.
"""

# Imports
from collections import defaultdict
from datetime import datetime, timedelta
from os.path import isfile
import joblib
import requests
import sys
from urllib.parse import urlparse
import yaml

from bs4 import BeautifulSoup
from matplotlib.dates import date2num, DateFormatter, DayLocator, MonthLocator
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

import access_classes as ac
import export_classes as ec

# Loading settings
# TODO integrate this in main() to allow calling main in notebook ?
# or not since settings can be set separately in notebook
with open("settings.yaml", "r") as settings_file:
    SETTINGS = yaml.safe_load(settings_file.read())
CMAP = getattr(plt.cm, SETTINGS['cmap'])

with open("author_convert.yaml", "r") as convert_file:
    CONVERT = yaml.safe_load(convert_file.read())

# Pre-declaring dict for selection of subclass of CommentThread
THREAD_TYPES = {}


# Main
def main(urls, do_more=True):
    """
    Creates thread based on supplied url(s), and tests some functionality.
    """
    filename = 'CACHE/' + SETTINGS['filename'] + "_" + 'mthread.p'
    if isfile(filename):
        print("loading {}:".format(filename), end=' ')
        an_mthread = joblib.load(filename)
        print("complete")
    else:
        the_threads = []
        print("Processing urls and creating {} threads".format(len(urls)))
        for url in urls:
            thread_type = urlparse(url).netloc.split('.')[0].title()
            print("processing {} as {}".format(url, thread_type))
            new_thread = THREAD_TYPES[thread_type](url)
            the_threads.append(new_thread)
        print("Merging threads in mthread:", end=' ')
        an_mthread = MultiCommentThread(*the_threads)
        print("complete")
        print("saving {} as {}:".format(type(an_mthread), filename), end=' ')
        joblib.dump(an_mthread, filename)
        print("complete")
    if do_more:
        # an_mthread.k_means()
        # return an_mthread
        an_mthread.draw_graph()
        an_mthread.plot_activity("author")
    else:
        return an_mthread


# Classes
class CommentThread(ac.ThreadAccessMixin, object):
    """
    Parent class for storing a comment thread to a WordPress
    post in a directed graph.
    Subclasses have the appropriate parsing method.
    Inherits methods from parent Mixin.

    Attributes:
        thread_url: parsed url (dict-like)
        _req: request from url.
        soup: BeautifullSoup parsing of content of request.
        post_title: title of post (only supplied by sub-classes).
        post_content: content of post (only supplied by sub-classes).
        graph: DiGraph based on thread (overruled from ThreadAccessMixin).
        node_name: dict with nodes as keys and authors as values
        authors: set with author-names

    Methods:
        parse_thread: not implemented.
        store_attributes: returns arguments in a dict
                          (called by child-classes)
        create_edges: takes graph and returns graph with edges added
                      (called by child classes)
        from ThreadAccessMixin:
            comment_report: takes node-id(s)
                            returns dict with report about node
            print_nodes: takes nodes-id(s),
                         prints out node-data as yaml
            tokenize_and_stem: tokenizes and stems com_content
                               (called as static method by store_attributes)
    """

    def __init__(self, url, comments_only):
        super(CommentThread, self).__init__()
        self.thread_url = urlparse(url)
        self._req = requests.get(url)
        self.soup = BeautifulSoup(self._req.content, SETTINGS['parser'])
        self.graph = self.parse_thread(self.soup, self.thread_url)
        self.post_title = ""
        self.post_content = ""
        # creates sub_graph and node:author dict based on comments_only
        if comments_only:
            # create node:name dict for nodes that are comments
            self.node_name = {node_id: data['com_author'] for (node_id, data)
                              in self.graph.nodes_iter(data=True) if
                              data['com_type'] == 'comment'}
            # create sub_graph based on keys of node_name
            self.graph = self.graph.subgraph(self.node_name.keys())
        else:
            self.node_name = nx.get_node_attributes(self.graph, 'com_author')
        self.authors = set(self.node_name.values())

    @classmethod
    def parse_thread(cls, a_soup, url):
        """Abstract method: raises NotImplementedError."""
        raise NotImplementedError("Subclasses should implement this!")

    @classmethod
    def store_attributes(cls,
                         com_class, com_depth, com_all_content, time_stamp,
                         com_author, com_author_url, child_comments,
                         thread_url):
        """Processes post-content, and returns arguments as dict"""
        content = " ".join(com_all_content)
        tokens, stems = ac.ThreadAccessMixin.tokenize_and_stem(content)
        return {"com_type": com_class[0],
                "com_depth": com_depth,
                "com_content": content,
                "com_tokens": tokens,
                "com_stems": stems,
                "com_timestamp": time_stamp,
                "com_author": com_author,
                "com_author_url": com_author_url,
                "com_children": child_comments,
                "com_thread": thread_url}

    @classmethod
    def create_edges(cls, a_graph):
        """
        Takes nx.DiGraph, adds edges to child_comments and returns nx.DiGraph.
        """
        for node_id, children in\
                nx.get_node_attributes(a_graph, "com_children").items():
            if children:
                a_graph.add_edges_from(((node_id,
                                         child) for child in children))
        return a_graph


class MultiCommentThread(ac.ThreadAccessMixin, ec.GraphExportMixin, object):
    """
    Combines graphs of multiple comment_threads for uniform author colouring.
    Main uses: drawing graphs, and supply to author_network.
    Drawing of separate threads should also use this class.
    Inherits methods from two parent Mixins.

    Attributes:
        graph: nx.DiGraph (overruled from ThreadAccessMixin).
        author_color: dict with authors as keys and colors (ints) as values.
        node_name: dict with nodes as keys and authors as values
        type_nodes: defaultdict with thread_class as key and
                   list of authors as values. CORRECT???
        thread_urls: list of urls of the respective threads
        corpus: list of unicode-strings (one per comment)
        vocab: dict with:
            tokenized: flat list of tokens (of all comments)
            stemmed: flat list of stems (of all comments)
            frame: pandas.DataFrame with vocab_tokenized as 'word' columns,
                     and vocab_stemmed as index

    Methods:
        add_thread: mutator method for adding thread to multithread.
                    This method is called by init.
        draw_graph: accessor method that draws the mthread graph.
        plot_activity: accessor method plotting of x-axis: time_stamps
                                                   y-axis: what's active
        tf_idf: accessor method that returns (tfidf_vectorizer, tfidf_matrix)
        from ThreadAccessMixin:
            comment_report: takes node-id(s), and
                            returns dict with report about node.
            print_nodes: takes nodes-id(s), and
                         prints out node-data as yaml. No output.
        from ThreadExportMixin:
            to_gephi: exports the full graph to gephi.
            to_yaml: exports the full graph to yaml.

    """

    def __init__(self, *threads):
        super(MultiCommentThread, self).__init__()
        self.author_color = {}
        self.node_name = {}
        self.type_nodes = defaultdict(list)
        self.thread_urls = []
        self.corpus = []
        self.vocab = {'tokenized': [],
                      'stemmed': []}
        # self.vocab_tokenized = []
        # self.vocab_stemmed = []
        for thread in threads:
            self.add_thread(thread, replace_frame=False)
            self.type_nodes[thread.__class__.__name__] += thread.graph.nodes()
        self.vocab['frame'] = DataFrame({'words': self.vocab['tokenized']},
                                        index=self.vocab['stemmed'])

    # Mutator methods
    def add_thread(self, thread, replace_frame=True):
        """
        Adds new (non-overlapping) thread by updating author_color and DiGraph.
        """
        # step 1: updating of lists and dicts
        new_authors = thread.authors.difference(self.author_color.keys()) # removed list
        new_colors = {a: c for (a, c) in
                      zip(new_authors,
                          range(len(self.author_color),
                                len(self.author_color) + len(new_authors)))}
        self.author_color.update(new_colors)
        # assert tests for non-overlap of node_id's between threads
        assert set(self.node_name.keys()).intersection(
            set(thread.node_name.keys())) == set([])
        self.node_name.update(thread.node_name)
        self.thread_urls.append(thread.thread_url)
        # step 2: updating vocabularies
        for _, data in thread.graph.nodes_iter(data=True):
            self.corpus.append(data["com_content"])
            self.vocab['tokenized'].extend(data["com_tokens"])
            self.vocab['stemmed'].extend(data["com_stems"])
        if replace_frame:  # only when called outside init
            self.vocab['frame'] = DataFrame({'words': self.vocab['tokenized']},
                                            index=self.vocab['stemmed'])
        # step 3: composing graphs
        self.graph = nx.compose(self.graph, thread.graph)

    # Accessor methods
    def draw_graph(self, time_intervals=5, show=True):
        """Draws and shows graph."""
        # creating title and axes
        title = "Thread structure for {}".format(SETTINGS['msg']).title()
        figure = plt.figure()
        figure.suptitle(title, fontsize=12)
        axes = figure.add_subplot(111)
        axes.yaxis.set_major_locator(DayLocator(interval=time_intervals))
        axes.yaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_ticks(range(1, 7))
        axes.set_xlabel("Comment Levels")
        # creating and drawingsub_graphs
        types_markers = {thread_type: marker for (thread_type, marker) in
                         zip(self.type_nodes.keys(),
                             ['o', '>', 'H', 'D'][:len(self.type_nodes.keys())]
                            )}
        for (thread_type, marker) in types_markers.items():
            type_subgraph = self.graph.subgraph(self.type_nodes[thread_type])
            # generating colours and positions for sub_graph
            positions = {node_id: (data["com_depth"],
                                   date2num(data["com_timestamp"]))
                         for (node_id, data) in
                         type_subgraph.nodes_iter(data=True)}
            node_color = {node_id: (self.author_color[self.node_name[node_id]])
                          for node_id in type_subgraph.nodes()}
            # drawing nodes of type_subgraph
            nx.draw_networkx_nodes(type_subgraph, positions,
                                   node_size=20,
                                   nodelist=node_color.keys(),
                                   node_color=node_color.values(),
                                   node_shape=marker,
                                   vmin=SETTINGS['vmin'],
                                   vmax=SETTINGS['vmax'],
                                   cmap=CMAP,
                                   ax=axes)
            nx.draw_networkx_edges(type_subgraph, positions, width=.5)
            if SETTINGS['show_labels_comments']:
                nx.draw_networkx_labels(
                    type_subgraph, positions, font_size=8,
                    labels={node: node[9:] for node in node_color.keys()})
        # show all
        plt.style.use('ggplot')
        the_lines = [mlines.Line2D([], [], color='gray',
                                   marker=marker,
                                   markersize=5,
                                   label=thread_type[13:])
                     for (thread_type, marker) in types_markers.items()]
        plt.legend(title="Where is the discussion happening",
                   handles=the_lines)
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def plot_activity(self, activity,
                      time_delta=timedelta(15),
                      max_span=timedelta(5000),
                      intervals=1,
                      show=True):
        """
        Shows plot of x-axis: time_stamps,
                      y-axis: what's active (author / thread)
        """
        start, stop = datetime(2016, 1, 1), datetime(2000, 1, 1)
        if activity.lower() == "author":
            items = list(self.author_color.keys())
            tick_tuple = tuple(items)
            key = "com_author"
        elif activity.lower() == "thread":
            items = self.thread_urls
            tick_tuple = tuple([item.netloc + "\n" + item.path for
                                item in items])
            key = "com_thread"
        else:
            pass
        for y_value, item in enumerate(items, start=1):
            norm = mpl.colors.Normalize(vmin=SETTINGS['vmin'],
                                        vmax=SETTINGS['vmax'])
            c_mp = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
            v_color = c_mp.to_rgba(self.author_color[item]) if\
                activity.lower() == "author" else 'k'
            timestamps = [data["com_timestamp"] for _, data in
                          self.graph.nodes_iter(data=True)
                          if data[key] == item]
            this_start, this_stop = min(timestamps), max(timestamps)
            start, stop = min(start, this_start), max(stop, this_stop)
            plt.hlines(y_value, this_start, this_stop, v_color, lw=.5)
            for timestamp in timestamps:
                plt.vlines(timestamp,
                           y_value+0.05, y_value-0.05,
                           v_color, lw=1)
        # Setup the plot
        plt.title("{} activity over time for {}".format(
            activity, SETTINGS['msg']).title(), fontsize=12)
        plt.style.use('ggplot')
        axes = plt.gca()
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(DateFormatter('%b %d, %Y'))
        axes.xaxis.set_major_locator(MonthLocator(interval=intervals))
        plt.xlim(start-time_delta,
                 min([stop+time_delta, start+max_span]))
        plt.yticks(range(1, len(items)+1), tick_tuple)
        if show:
            plt.show()
        else:
            filename = input("Give filename: ")
            filename += ".png"
            plt.savefig(filename)

    def tf_idf(self):
        """Initial tf_idf method (incomplete)"""
        obj_names = ['tfidf_vectorizer',
                     'tfidf_matrix',
                     'tfidf_terms',
                     'tfidf_dist']
        filenames = ['CACHE/' + SETTINGS['filename'] +
                     "_" + obj_name + ".p" for obj_name in obj_names]
        objs = []
        # create vectorizer (cannot be pickled)
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                           min_df=0.0, stop_words='english',
                                           use_idf=True,
                                           tokenizer=lambda text:
                                           ac.ThreadAccessMixin.
                                           tokenize_and_stem(text)[1],
                                           ngram_range=(1, 3))
        objs.append(tfidf_vectorizer)
        # check for pickled objects (all but first in list)
        if isfile(filenames[1]):
            # loading and adding to list
            for filename in filenames[1:]:
                print("Loading {}: ".format(filename), end=' ')
                objs.append(joblib.load(filename))
                print("complete")
        else:  # create and pickle
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.corpus)
            objs.append(tfidf_matrix)
            tfidf_terms = tfidf_vectorizer.get_feature_names()
            objs.append(tfidf_terms)
            tfidf_dist = 1 - cosine_similarity(tfidf_matrix)
            objs.append(tfidf_dist)
            for filename, obj_name, obj in zip(filenames, obj_names, objs)[1:]:
                print("Saving {} as {}: ".format(obj_name, filename), end=' ')
                joblib.dump(obj, filename)
                print("complete")
        return {name: obj for (name, obj) in zip(obj_names, objs)}

    def k_means(self, num_clusters=5, num_words=15, reset=False):
        """k_means"""
        # assigning from tfidf
        matrix, terms, dist = (self.tf_idf()['tfidf_matrix'],
                               self.tf_idf()['tfidf_terms'],
                               self.tf_idf()['tfidf_dist'])
        filename = 'CACHE/' + SETTINGS['filename'] + "_" + 'kmeans.p'
        if isfile(filename) and not reset:
            print("Loading kmeans: ", end=' ')
            kmeans = joblib.load(filename)
            print("complete")
        else:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(matrix)
            print("Saving kmeans: ", end=' ')
            joblib.dump(kmeans, filename)
            print("complete")
        clusters = kmeans.labels_.tolist()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        nodes, times, authors, blog = list(zip(*[
            (node, data["com_timestamp"],
             data["com_author"].encode("utf-8"),
             data["com_thread"].netloc[:-14])
            for (node, data) in self.graph.nodes_iter(data=True)]))
        comments = {'com_id': list(nodes),
                    'time_stamps': list(times),
                    'com_authors': list(authors),
                    'blog': list(blog),
                    'cluster': clusters}
        frame = DataFrame(comments,
                          index=[clusters],
                          columns=['com_id',
                                   'time_stamps',
                                   'com_authors',
                                   'blog',
                                   'cluster'])
        print()
        print("Top terms per cluster:")
        print()
        for i in range(num_clusters):
            print("Cluster {} size: {}".format(i,
                                               len(frame.ix[i]['com_id'].
                                                   values.tolist())))
            print("Cluster {} words: ".format(i), end=' ')
            for ind in order_centroids[i, :num_words]:
                print(self.vocab['frame'].ix[terms[ind].split(' ')].\
                      values.tolist()[0][0].encode('utf-8', 'ignore'), end=' ')
            print("\n", end=' ')
            print("Cluster {} authors: ".format(i), end=' ')
            for author in set(frame.ix[i]['com_authors'].values.tolist()):
                print("{}".format(author), end=' ')
            print()
            print()
        # multi dimensional scaling
        MDS()
        print("assigning mds")
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
        print("fitting: ", end=' ')
        pos = mds.fit_transform(dist)
        print("complete")
        xs, ys = pos[:, 0], pos[:, 1]
        cluster_colors = {0: '#1b9e77',
                          1: '#d95f02',
                          2: '#7570b3',
                          3: '#e7298a',
                          4: '#66a61e'}
        cluster_names = {0: 'one',
                         1: 'two',
                         2: 'three',
                         3: 'four',
                         4: 'five'}
        df = DataFrame(dict(x=xs, y=ys, label=clusters, title=nodes))
        groups = df.groupby('label')
        _, ax = plt.subplots(figsize=(17, 9))
        ax.margins(0.05)
        # iterate through groups to layer the plot
        # note that I use the cluster_name and cluster_color dicts with the
        # 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='',
                    ms=12, label=cluster_names[name],
                    color=cluster_colors[name], mec='none')
            ax.set_aspect('auto')
            ax.tick_params(axis='x',
                           which='both',
                           bottom='off',
                           top='off',
                           labelbottom='off')
            ax.tick_params(axis='y',
                           which='both',
                           left='off',
                           top='off',
                           labelleft='off')

        ax.legend(numpoints=1)  # show legend with only 1 point

        # add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'],
                    df.ix[i]['y'],
                    df.ix[i]['title'],
                    size=8)

        plt.show()  # show the plot


class CommentThreadPolymath(CommentThread):
    """ Child class for PolyMath Blog, with method for actual parsing. """
    def __init__(self, url, comments_only=True):
        super(CommentThreadPolymath, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post"}).find("h3").text
        self.post_content = self.soup.find(
            "div", {"class": "storycontent"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"id": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word
                             in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find(
                                   "div", {"class": "comment-author vcard"}
                                   ).find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").find("span").text
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp (and time is poped from all_content)
            time_stamp = " ".join(com_all_content.pop().split()[-7:])[2:]
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%B %d, %Y @ %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find(
                    "a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find(
                    "ul", {"class": "children"}).find_all(
                        "li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadGilkalai(CommentThread):
    """ Child class for Gil Kalai Blog, with method for actual paring. """
    def __init__(self, url, comments_only=True):
        super(CommentThreadGilkalai, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"id": "content"}).find("h2", {"class": "entry-title"}).text
        self.post_content = self.soup.find(
            "div", {"class": "entry-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"class": "commentlist"})
        if the_comments:
            # NOTE: Pingbacks have no id and are ignored
            all_comments = the_comments.find_all("li", {"class": "comment"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.find("div").get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if
                             word.startswith("depth-"))
            com_all_content = [item.text for item in comment.find(
                "div", {"class": "comment-body"}).find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").text.strip()
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find("div",
                                      {"class": "comment-meta commentmetadata"}
                                     ).text.strip()
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%B %d, %Y at %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find(
                    "a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find(
                    "ul", {"class": "children"}).find_all(
                        "li", {"class": depth_search})
                child_comments = [child.find("div").get("id") for
                                  child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadGowers(CommentThread):
    """ Child class for Gowers Blog, with method for actual paring."""
    def __init__(self, url, comments_only=True):
        super(CommentThreadGowers, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post"}).find("h2").text
        self.post_content = self.soup.find(
            "div", {"class": "post"}).find(
                "div", {"class": "entry"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("ol", {"class": "commentlist"})
        if the_comments:
            all_comments = the_comments.find_all("li")
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for
                             word in com_class if word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find_all("p")]
            # getting and converting author_name
            try:
                com_author = comment.find("cite").text.strip()
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find("small").find("a").text
            try:
                time_stamp = datetime.strptime(
                    time_stamp, "%B %d, %Y at %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find("cite").find(
                    "a", {"rel": "external nofollow"}).get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                child_comments = comment.find(
                    "ul", {"class": "children"}).find_all(
                        "li", {"class": depth_search})
                child_comments = [child.get("id") for child in child_comments]
            except AttributeError:
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph


class CommentThreadTerrytao(CommentThread):
    """ Child class for Tao Blog, with method for actual paring."""
    def __init__(self, url, comments_only=True):
        super(CommentThreadTerrytao, self).__init__(url, comments_only)
        self.post_title = self.soup.find(
            "div", {"class": "post-meta"}).find("h1").text
        self.post_content = self.soup.find(
            "div", {"class": "post-content"}).find_all("p")

    @classmethod
    def parse_thread(cls, a_soup, thread_url):
        """
        Creates and returns an nx.DiGraph from the comment_soup.
        This method is only used by init.
        """
        a_graph = nx.DiGraph()
        the_comments = a_soup.find("div", {"class": "commentlist"})
        if the_comments:
            # this seems to ignore the pingbacks
            all_comments = the_comments.find_all("div", {"class": "comment"})
            all_comments += the_comments.find_all("div", {"class": "pingback"})
        else:
            all_comments = []  # if no commentlist found
        for comment in all_comments:
            # identify id, class, depth and content
            com_id = comment.get("id")
            com_class = comment.get("class")
            com_depth = next(int(word[6:]) for word in com_class if
                             word.startswith("depth-"))
            com_all_content = [item.text for item in
                               comment.find_all("p")][2:]
            # getting and converting author_name
            try:
                com_author = comment.find(
                    "p", {"class": "comment-author"}).text
            except AttributeError as err:
                print(err, comment.find("cite"))
                com_author = "unable to resolve"
            com_author = CONVERT[com_author] if\
                com_author in CONVERT else com_author
            # creating timeStamp
            time_stamp = comment.find(
                "p", {"class": "comment-permalink"}).find("a").text
            try:
                time_stamp = datetime.strptime(time_stamp,
                                               "%d %B, %Y at %I:%M %p")
            except ValueError as err:
                print(err, ": datetime failed")
            # getting href to comment author webpage (if available)
            try:
                com_author_url = comment.find(
                    "p", {"class": "comment-author"}).find("a").get("href")
            except AttributeError:
                com_author_url = None
            # make list of child-comments (only id's)
            try:
                depth_search = "depth-" + str(com_depth+1)
                if comment.next_sibling.next_sibling['class'] == ['children']:
                    child_comments = comment.next_sibling.next_sibling.\
                                    find_all("div",
                                             {"class": "comment"})
                    child_comments = [child.get("id") for child in
                                      child_comments if depth_search in
                                      child["class"]]
                else:
                    child_comments = []
            except (AttributeError, TypeError):
                child_comments = []
            # creating dict of comment properties as attributes of nodes
            attr = cls.store_attributes(com_class,
                                        com_depth,
                                        com_all_content,
                                        time_stamp,
                                        com_author,
                                        com_author_url,
                                        child_comments,
                                        thread_url)
            # adding node
            a_graph.add_node(com_id)
            # adding all attributes to node
            for (key, value) in attr.items():
                a_graph.node[com_id][key] = value
        # creating edges
        a_graph = cls.create_edges(a_graph)
        return a_graph

THREAD_TYPES = {"Polymathprojects": CommentThreadPolymath,
                "Gilkalai": CommentThreadGilkalai,
                "Gowers": CommentThreadGowers,
                "Terrytao": CommentThreadTerrytao}

if __name__ == '__main__':
    ARGUMENTS = sys.argv[1:]
    if ARGUMENTS:
        SETTINGS['filename'] = input("Filename to be used: ")
        SETTINGS['msg'] = input("Message to be used: ")
        main(ARGUMENTS)
    else:
        print(SETTINGS['msg'])
        main(SETTINGS['urls'])
