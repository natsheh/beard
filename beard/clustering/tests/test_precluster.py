# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of precluster class.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import pytest

from ..preclustering import Precluster


@pytest.fixture
def precluster():
    """Create a precluster for mr Abc, D. Vasquez."""
    return Precluster((("ABC",), ("D", "VSQ")))


def test_add_signature(precluster):
    """Test adding signatures to the cluster."""
    assert precluster._content[("ABC",)][("D", "VSQ")] == 1
    precluster.add_signature((("ABC",), ("D", "VSQ")))
    assert precluster._content[("ABC",)][("D", "VSQ")] == 2
    precluster.add_signature((("ABC",), ("E",)))
    assert precluster._content[("ABC",)][("E",)] == 1
    precluster.add_signature((("ABD",), ("D", "VSQ",)))
    assert precluster._content[("ABD",)][("D", "VSQ")] == 1
    precluster.add_signature((("ABC", ""), ("D", "VSQ")))
    # Check handling of multiple surnames
    precluster.add_signature((("ABD", "EFG"), ("D", "VSQ",)))
    assert precluster._content[("ABD", "EFG")][("D", "VSQ")] == 1
    assert precluster._content[("ABC",)][("D", "VSQ")] == 2


def test_compare_tokens_from_back(precluster):
    """Test comparing tokens from the back."""
    assert precluster.compare_tokens_from_back(("VSQ",), ("ABC",))
    assert precluster.compare_tokens_from_back(("C", "D", "VSQ",), ("ABC",))
    with pytest.raises(KeyError) as excinfo:
        precluster.compare_tokens_from_back(("VSQ",), ("DEF"))
    assert "cluster doesn't contain a key" in str(excinfo.value)
    assert not precluster.compare_tokens_from_back(("VSD",), ("ABC",))
    assert not precluster.compare_tokens_from_back(("DGM", "VSQ"), ("ABC",))


def test_initials_score(precluster):
    """Test initial scoring function."""
    precluster.add_signature((("ABC",), ("D", "VSQ")))
    assert precluster.initials_score(("D",), ("ABC",)) == 2
    assert precluster.initials_score(("D", "V"), ("ABC",)) == 2
    assert precluster.initials_score(("V"), ("ABC",)) == 2
    assert precluster.initials_score(("D", "VSQ"), ("ABC",)) == 2
    assert precluster.initials_score(("D", "V", "R"), ("ABC",)) == 0
    assert precluster.initials_score(("D", "VR"), ("ABC",)) == 0
    precluster.add_signature((("", ""), ("ABC",)))
    assert precluster.initials_score(("",), ("ABC",)) == 1
    assert precluster.initials_score(("H",), ("ABC",)) == 0

    # Check wrong key
    with pytest.raises(KeyError) as excinfo:
        precluster.initials_score(("VSQ",), ("DEF"))
    assert "cluster doesn't contain a key" in str(excinfo.value)

    # Double metaphone algorithm can output empty string as the result
    precluster.add_signature((("ABC",), ("E", "")))
    assert precluster.initials_score(("E", "A"), ("ABC",)) == 1

    # Test the correctness of the function for multiple entities in the
    # precluster
    precluster.add_signature((("ABC",), ("D")))
    assert precluster.initials_score(("D", "V"), ("ABC",)) == 2
    assert precluster.initials_score(("D",), ("ABC",)) == 3


def test_single_names_variants():
    """Test retrieving the number of signature."""
    precluster_ = Precluster((("ABC",), ("D", "VSQ")))
    assert precluster_.single_names_variants() == 1
    precluster_.add_signature((("ABC",), ("D", "VSQ")))
    assert precluster_.single_names_variants() == 2
    precluster_.add_signature((("ABC", "DEF"), ("D", "VSQ")))
    assert precluster_.single_names_variants() == 2
    precluster_ = Precluster((("ABC", "DEF"), ("D", "VSQ")))
    assert precluster_.single_names_variants() == 1
    precluster_.add_signature((("ABC", "DEF"), ("D", "VSQ")))
    assert precluster_.single_names_variants() == 1


def test_contains(precluster):
    """Test contains method."""
    assert precluster.contains(("ABC",))
    assert not precluster.contains(("DEF",))
