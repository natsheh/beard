# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of the preclustering algorithm.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import numpy as np

from ..preclustering import dm_preclustering


def run_preclustering(names, expected_results, threshold=100):
    """Run dm_preclustering and assert that the results are correct."""
    sigs = np.array([{'author_name': sig} for sig in names])
    for index, value in enumerate(dm_preclustering(sigs, threshold)):
        assert names[index] == expected_results[index]


def test_single_signature():
    """Cluster one signature."""
    run_preclustering(['Smith, Joe'], ['SM0'])


def test_first_surname_included():
    """Check first surname full match."""
    run_preclustering(['Smith-Jones, Joe', 'Smith, Joe',
                       'Jones, Paul', 'Smith-Jones, Paul'],
                      ['SM0', 'SM0', 'JNS', 'SM0'])


def test_last_surname_included():
    """Check last surname full match and match by token."""
    run_preclustering(['Jones-Smith, Joe', 'Smith, Joe',
                       'Jones, Paul', 'Jones-Smith, Paul'],
                      ['SM0', 'SM0', 'JNS', 'SM0'])


def test_no_suitable_block_for_multiple_surnames():
    """Check if a block is created for surnames that don't match."""
    run_preclustering(['Jones-Smith, Joe'], ['JNS'])


def test_precluster_splitt():
    """Check if huge blocks are split."""
    run_preclustering(['Smith, Joe', 'Smith, Paul'], ['SM0j', 'SM0p'])
