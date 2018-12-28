import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import math
# OUTPUT
# [[4.8   3.05 ]
# [5.3   4.   ]
# [6.2   3.025]]

a2 = [1.0, 1.0]
a3 = [5.0, 7.0]

X = [
[1.0,   1.0],
[1.5,	2.0],
[3.0,	4.0],
[5.0,	7.0],
[3.5,	5.0],
[4.5,	5.0],
[3.5,	4.5],
]

X = np.asarray(X)
init_center = np.asarray([a2, a3])

# kmeans = KMeans(init = init_center, n_clusters = 3)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)
# print(kmeans.cluster_centers_)

class KMeans:
    def __init__(self, k , init_center = np.asarray([])):
        self.k = k
        self.point_cluster = []
        if init_center.shape[0] > 0 and init_center.shape[0] == k:
            self.centroids = init_center
        else:
            self.centroids = np.zeros((k, 2))  ###  X , Y, number_of_entries

    def euclidean_distance(self, X, Y):
        tmp = X - Y
        sum_squared = np.linalg.norm(tmp)
        return sum_squared

    def calculate_which_centroid(self, ELEM):
        min_eu = self.euclidean_distance(self.centroids[0], ELEM)
        min_index = 0
        for i in range(1, self.k):
            centroid = self.centroids[i]
            eu = self.euclidean_distance(centroid, ELEM)
            if eu <= min_eu:
                min_index = i
                min_eu = eu

        self.point_cluster.append(min_index)

    def update_centroid(self, X):
        self.counter_centroid = defaultdict(lambda:0)
        print(self.point_cluster)
        for i in range(len(self.point_cluster)):
            self.centroids[self.point_cluster[i]] += X[i]
            self.counter_centroid[self.point_cluster[i]] += 1

        print(self.centroids)
        print(self.counter_centroid)
        for i in range(self.k):
            if self.counter_centroid[i] != 0:
                self.centroids[i] /= self.counter_centroid[i]

    def fit(self, X, EPOCHS):
        for i in range(EPOCHS):
            self.point_cluster = []
            for x in X:
                self.calculate_which_centroid(x)
            #print(self.centroids)
            self.update_centroid(X)
            #print(self.centroids)
        return self.centroids

k = KMeans(2, init_center)
cluster_centers_ = k.fit(X, 1)
