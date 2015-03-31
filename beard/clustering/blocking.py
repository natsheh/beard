# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Blocking for clustering estimators.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

from __future__ import print_function

import numpy as np

from joblib import delayed
from joblib import Parallel

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import ClusterMixin
from sklearn.utils import column_or_1d


def _single(X):
    return np.zeros(len(X), dtype=np.int)


class _SingleClustering(BaseEstimator, ClusterMixin):
    def fit(self, X, y=None):
        self.labels_ = _single(X)
        return self

    def partial_fit(self, X, y=None):
        self.labels_ = _single(X)
        return self

    def predict(self, X):
        return _single(X)


def _parallel_fit(fit_, partial_fit_, b, X, y, clusterer, verbose):
    """Run clusterer's fit function."""
    if verbose > 1:
        print("Clustering %d samples on block '%s'..." % (len(X), b))

    if fit_ or not hasattr(clusterer, "partial_fit"):
        try:
            clusterer.fit(X, y=y)
        except TypeError:
            clusterer.fit(X)
    elif partial_fit_:
        try:
            clusterer.partial_fit(X, y=y)
        except TypeError:
            clusterer.partial_fit(X)

    return (b, clusterer)


class BlockClustering(BaseEstimator, ClusterMixin):

    """Implements blocking for clustering estimators.

    Meta-estimator for grouping samples into blocks, within each of which
    a clustering base estimator is fit. This allows to reduce the cost of
    pairwise distance computation from O(N^2) to O(sum_b N_b^2), where
    N_b <= N is the number of samples in block b.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    blocks_ : ndarray, shape (n_samples,)
        Array of keys mapping input data to blocks.
    """

    def __init__(self, affinity=None, blocking="single", base_estimator=None,
                 verbose=0, n_jobs=1):
        """Initialize.

        Parameters
        ----------
        :param affinity: string or None
            If affinity == 'precomputed', then assume that X is a distance
            matrix.

        :param blocking: string or callable, default "single"
            The blocking strategy, for mapping samples X to blocks.
            - "single": group all samples X[i] into the same block;
            - "precomputed": use `blocks[i]` argument (in `fit`, `partial_fit`
              or `predict`) as a key for mapping sample X[i] to a block;
            - callable: use blocking(X)[i] as a key for mapping sample X[i] to
              a block.

        :param base_estimator: estimator
            Clustering estimator to fit within each block.

        :param verbose: int, default=0
            Verbosity of the fitting procedure.

        :param n_jobs: int
            Parameter passed directly to joblib library.
        """
        self.affinity = affinity
        self.blocking = blocking
        self.base_estimator = base_estimator
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _validate(self, X, blocks):
        """Validate hyper-parameters and input data."""
        if self.blocking == "single":
            blocks = _single(X)
        elif self.blocking == "precomputed":
            if blocks is not None and len(blocks) == len(X):
                blocks = column_or_1d(blocks).ravel()
            else:
                raise ValueError("Invalid value for blocks. When "
                                 "blocking='precomputed', blocks needs to be "
                                 "an array of size len(X).")
        elif callable(self.blocking):
            blocks = self.blocking(X)
        else:
            raise ValueError("Invalid value for blocking. Allowed values are "
                             "'single', 'precomputed' or callable.")

        return X, blocks

    def _blocks(self, X, y, blocks):
        """Chop the training data into smaller chunks.

        A chunk is demarcated by the corresponding block. Each chunk contains
        only the training examples relevant to given block and a clusterer
        which will be used to fit the data.

        Returns
        -------
        :returns: generator
            Quadruples in the form of ``(block, X, y, clusterer)`` where
            X and y are the training examples for given block and clusterer is
            an object with a ``fit`` method.
        """
        unique_blocks = np.unique(blocks)

        for b in unique_blocks:
            mask = (blocks == b)
            X_mask = X[mask, :]
            if y is not None:
                y_mask = y[mask]
            else:
                y_mask = None
            if self.affinity == "precomputed":
                X_mask = X_mask[:, mask]

            # Select a clusterer
            if len(X_mask) == 1:
                clusterer = _SingleClustering()
            elif self.fit_:
                # Although every job has its own copy of the estimator, one job
                # can serve multiple fits. That's why the clone is needed.
                clusterer = clone(self.base_estimator)
            elif self.partial_fit_:
                if b in self.clusterers_:
                    clusterer = self.clusterers_[b]
                else:
                    clusterer = clone(self.base_estimator)

            yield (b, X_mask, y_mask, clusterer)

    def _fit(self, X, y, blocks):
        """Fit base clustering estimators on X."""
        self.blocks_ = blocks

        results = (Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
                   (delayed(_parallel_fit)(self.fit_, self.partial_fit_,
                                           b, X_mask, y_mask, clusterer,
                                           self.verbose) for
                   b, X_mask, y_mask, clusterer in self._blocks(X, y, blocks)))

        for b, clusterer in results:
            self.clusterers_[b] = clusterer

        return self

    def fit(self, X, y=None, blocks=None):
        """Fit individual base clustering estimators for each block.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
                  or (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Reset attributes
        self.clusterers_ = {}
        self.fit_, self.partial_fit_ = True, False

        return self._fit(X, y, blocks)

    def partial_fit(self, X, y=None, blocks=None):
        """Resume fitting of base clustering estimators, for each block.

        This calls `partial_fit` whenever supported by the base estimator.
        Otherwise, this calls `fit`, on given blocks only.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
                  or (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Set attributes if first call
        if not hasattr(self, "clusterers_"):
            self.clusterers_ = {}

        self.fit_, self.partial_fit_ = False, True

        return self._fit(X, y, blocks)

    def predict(self, X, blocks=None):
        """Predict data.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: array-like, shape (n_samples)
            The labels.
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Predict
        labels = -np.ones(len(X), dtype=np.int)
        offset = 0

        for b in np.unique(blocks):
            # Predict on the block, if known
            if b in self.clusterers_:
                mask = (blocks == b)
                clusterer = self.clusterers_[b]

                pred = np.array(clusterer.predict(X[mask]))
                pred[(pred != -1)] += offset
                labels[mask] = pred
                offset += np.max(clusterer.labels_) + 1

        return labels

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly.
        """
        labels = -np.ones(len(self.blocks_), dtype=np.int)
        offset = 0

        for b in self.clusterers_:
            mask = (self.blocks_ == b)
            clusterer = self.clusterers_[b]

            pred = np.array(clusterer.labels_)
            pred[(pred != -1)] += offset
            labels[mask] = pred
            offset += np.max(clusterer.labels_) + 1

        return labels
