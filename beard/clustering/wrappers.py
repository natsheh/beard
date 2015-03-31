# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Scikit-Learn compatible wrappers of clustering algorithms.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Hussein AL-NATSHEH <h.natsheh@ciapple.com>

"""
import numpy as np

import scipy.cluster.hierarchy as hac

from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from beard.metrics import paired_f_score
from beard.metrics import b3_f_score


class ScipyHierarchicalClustering(BaseEstimator, ClusterMixin):

    """Wrapper for Scipy's hierarchical clustering implementation.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.

    linkage_ : ndarray
        The linkage matrix.
    """

    def __init__(self, method="single", affinity="euclidean",
                 threshold=None, n_clusters=1, criterion="distance",
                 depth=2, R=None, monocrit=None, scoring=b3_f_score,
                 affinity_score=False):
        """Initialize.

        Parameters
        ----------
        :param method: string
            The linkage algorithm to use.
            See scipy.cluster.hierarchy.linkage for further details.

        :param affinity: string or callable
            The distance metric to use.
            - "precomputed": assume that X is a distance matrix;
            - callable: a function returning a distance matrix;
            - Otherwise, any value supported by
              scipy.cluster.hierarchy.linkage.

        :param threshold: float or None
            The thresold to apply when forming flat clusters. In case
            of semi-supervised clustering, this value is overridden by
            the threshold maximizing the provided scoring function on
            the labeled samples.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param n_clusters: int
            The number of flat clusters to form, if threshold=None.

        :param criterion: string
            The criterion to use in forming flat clusters.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param depth: int
            The maximum depth to perform the inconsistency calculation.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param R: array-like or None
            The inconsistency matrix to use for the 'inconsistent' criterion.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param monocrit: array-like or None
            The statistics upon which non-singleton i is thresholded.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param scoring: callable
            The scoring function to maximize in either (semi-)supervised
            clustering if y is not None, or unsupervised otherwise (using
            "silhouette_score as an example when y is None or all y =-1).
            In case of silhouette_score, the metric param of this function
            must be set to 'precomputed'.
            See sklearn.metrics.silhouette_score for further details.

        :param affinity_score: boolean
            A flag that must be Ture if the scoring function requiers the
            affinity as an input. False otherise.
            )
        """
        self.method = method
        self.affinity = affinity
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.criterion = criterion
        self.depth = depth
        self.R = R
        self.monocrit = monocrit
        self.scoring = scoring
        self.affinity_score = affinity_score

    def fit(self, X, y=None):
        """Perform hierarchical clustering on input data.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features) or
                  (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        Returns
        -------
        :returns: self
        """
        X = np.array(X)

        # Build linkage matrix
        if self.affinity == "precomputed":
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)

        elif callable(self.affinity):
            X = self.affinity(X)
            Xs = X  # to be used in silhouette case
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)
        else:
            self.linkage_ = hac.linkage(X,
                                        method=self.method,
                                        metric=self.affinity)

        # Estimate threshold in case of semi-supervised or unsupervised

        if y is not None:
            y_arr = np.array(y)
            all_y_neg = y_arr.sum() == len(y_arr) * -1
            ground_truth = y is not None and not all_y_neg
        else:
            ground_truth = False

        if self.threshold is None:
            best_threshold = self.linkage_[-1, 2]
            best_score = -np.inf

            thresholds = np.concatenate(([0],
                                         self.linkage_[:, 2],
                                         [self.linkage_[-1, 2]]))

            for i in range(len(thresholds) - 1):
                t1, t2 = thresholds[i:i + 2]
                threshold = (t1 + t2) / 2.0
                labels = hac.fcluster(self.linkage_, threshold,
                                      criterion=self.criterion,
                                      depth=self.depth, R=self.R,
                                      monocrit=self.monocrit)
                if ground_truth:
                    train = (y != -1)

                    if train.sum() == 0:
                        return self

                    if not self.affinity_score:
                        score = self.scoring(y[train], labels[train])
                    else:
                        score = self.scoring(y[train], labels[train], Xs)

                elif self.affinity_score:
                    n_labels = len(np.unique(labels))
                    n_samples = Xs.shape[0]

                    if 1 < n_labels < n_samples:
                        score = self.scoring(Xs, labels)
                    else:
                        score = -np.inf
                else:
                    # could have a scoring function that accepts only lables
                    return self

                if score >= best_score:
                    best_score = score
                    best_threshold = threshold

            self.best_threshold_ = best_threshold

        return self

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly from the linkage matrix,
        based on the value of self.threshold or self.n_clusters.
        """
        threshold = self.threshold

        # Override default threshold with the estimated one
        if hasattr(self, "best_threshold_") and threshold is None:
            threshold = self.best_threshold_

        if threshold is not None:
            labels = hac.fcluster(self.linkage_, threshold,
                                  criterion=self.criterion, depth=self.depth,
                                  R=self.R, monocrit=self.monocrit)

            _, labels = np.unique(labels, return_inverse=True)
            return labels

        else:
            thresholds = np.concatenate(([0],
                                         self.linkage_[:, 2],
                                         [self.linkage_[-1, 2]]))

            for i in range(len(thresholds) - 1):
                t1, t2 = thresholds[i:i + 2]
                threshold = (t1 + t2) / 2.0
                labels = hac.fcluster(self.linkage_, threshold,
                                      criterion=self.criterion,
                                      depth=self.depth, R=self.R,
                                      monocrit=self.monocrit)

                if len(np.unique(labels)) == self.n_clusters:
                    _, labels = np.unique(labels, return_inverse=True)
                    return labels

            raise ValueError("Failed to group samples into n_clusters=%d"
                             % self.n_clusters)
