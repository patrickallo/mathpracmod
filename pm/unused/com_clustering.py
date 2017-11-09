"""
Module for mixin classes for comment-clustering methods
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS


class ClusteringMixin(object):
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
            for filename, obj_name, obj in list(
                zip(filenames, obj_names, objs))[1:]:
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