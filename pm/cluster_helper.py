"""
Module with helper-class to cluster timestamps collected from nx.graph
"""

# Imports
import logging

import numpy as np
from pandas import DataFrame
from sklearn.cluster import DBSCAN


class ClusterNodes(object):
    """Class that takes a graph, collected timestamp data from the nodes,
    clusters the timestamps, and stores DataFrame with cluster_labels
    for each node"""

    def __init__(self, a_graph, post_title):
        self.__handle_cluster_data(a_graph)
        self.__cluster_timestamps(post_title)
        self.__add_weights()
        self.__clean_outliers()

    def __handle_cluster_data(self, graph):
        self.node_data_ = dict((node, {'timestamps': data['com_timestamp'],
                                       'authors': data['com_author']})
                               for node, data in graph.nodes(data=True))
        try:
            data = DataFrame(self.node_data_).T.sort_values('timestamps')
        except KeyError:
            print(self.node_data_)
        for node in data.index:
            try:
                assert data.loc[node, 'timestamps'] == graph.node[
                    node]['com_timestamp']
            except AssertionError:
                print("Mismatch for ", node)
        epoch = data.ix[0, 'timestamps']
        data['timestamps'] = (data['timestamps'] - epoch).astype(int)
        self.data_ = data

    def __cluster_timestamps(self, post_title):
        cluster_data = self.data_['timestamps'].as_matrix().reshape(-1, 1)
        one_day = 86400000000000  # timedelta(1) to int to match data
        times_db = DBSCAN(eps=one_day / 2,
                          min_samples=2,
                          metric="euclidean")
        labels = times_db.fit_predict(cluster_data)
        unique_labels = np.sort(np.unique(labels))
        logging.info("Found %i clusters in %s",
                     len(unique_labels) - 1, post_title)
        try:
            assert len(labels) == len(cluster_data)
        except AssertionError as err:
            logging.warning("Mismatch cluster-labels: %s", err)
            print(unique_labels)
            print(labels)
        self.data_['cluster_id'] = labels

    def __add_weights(self):
        comments_per_cluster = self.data_['cluster_id'].value_counts()
        self.data_['weight'] = [comments_per_cluster[cluster] for
                                cluster in self.data_['cluster_id']]
        a_weights = self.data_.groupby(
            ['cluster_id', 'authors']).count()['timestamps']
        self.data_['author_weight'] = self.data_.apply(
            lambda x: a_weights[x['cluster_id'], x['authors']], axis=1)

    def __clean_outliers(self):
        self.data_ = self.data_.replace(
            to_replace={'cluster_id': {-1: np.nan}})
        self.data_.loc[np.isnan(self.data_['cluster_id']), 'weight'] = 0
        self.data_.loc[np.isnan(self.data_['cluster_id']), 'author_weight'] = 0
