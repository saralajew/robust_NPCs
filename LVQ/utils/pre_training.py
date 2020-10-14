# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import warnings

try:
    from sklearn.cluster import KMeans as sk_KMeans
except ImportError:
    sk_KMeans = None

try:
    from sklearn.cluster import MiniBatchKMeans as sk_MiniBatchKMeans
except ImportError:
    sk_MiniBatchKMeans = None

try:
    from sklearn.decomposition import TruncatedSVD as sk_TruncatedSVD
except ImportError:
    sk_TruncatedSVD = None


def kmeans(points, n_clusters, batch_version=False, **kmeans_params):
    def fit(x, **kmeans_params_):
        if batch_version is False:
            try:
                model = sk_KMeans(**kmeans_params_).fit(x)
            except MemoryError:
                warnings.warn("An MemoryError occurred during the execution of k-Means non-batch version. Be careful "
                              "that all provides parameters via `kmeans_params` are accept by the batch version and "
                              "have the desired value.")
                model = sk_MiniBatchKMeans(**kmeans_params_).fit(x)
        else:
            model = sk_MiniBatchKMeans(**kmeans_params_).fit(x)

        return model.cluster_centers_, model.labels_

    if sk_KMeans is None or sk_MiniBatchKMeans is None:
        raise ImportError('`pre_training` requires sklearn.')

    if not isinstance(points, (list, tuple)):
        points = [points]

    if not isinstance(n_clusters, (list, tuple)):
        n_clusters = np.repeat(n_clusters, len(points))
    elif len(n_clusters) != len(points):
        raise TypeError("n_clusters not understood. Provide n_clusters as int or list with len(points).")

    labels = []
    clusters = []
    for i, x in enumerate(points):
        kmeans_params.update({'n_clusters': n_clusters[i]})
        cluster, label = fit(x, **kmeans_params)
        clusters.append(cluster)
        labels.append(label)

    return clusters, labels


def svd(clusters,
        n_components,
        **svd_params):
    if sk_TruncatedSVD is None:
        raise ImportError('`pre_training` requires sklearn.')

    if not isinstance(clusters, (list, tuple)):
        clusters = [clusters]

    if not isinstance(n_components, (list, tuple)):
        n_components = np.repeat(n_components, len(clusters))
    elif len(n_components) != len(clusters):
        raise TypeError("n_components not understood. Provide n_components as int or list with len(clusters).")

    matrices = []
    valid = []
    for i, c in enumerate(clusters):
        # test if there are enough points in the cluster to compute a svd with the respective number of components
        if n_components[i] < c.shape[0]:
            # truncatedSVD doesn't work for n_components equal number of dimensions
            if n_components[i] == c.shape[-1]:
                matrix = np.linalg.eig(np.matmul(c.transpose(), c))[1]
            else:
                svd_params.update({'n_components': n_components[i]})
                model = sk_TruncatedSVD(**svd_params)
                model.fit(c)
                matrix = model.components_.transpose()
            matrices.append(matrix)
            valid.append(True)
        else:
            matrices.append(None)
            valid.append(False)

    return matrices, valid
