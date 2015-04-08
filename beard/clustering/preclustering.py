# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""The algorithm for preclustering.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

from copy import copy

import numpy as np
import six

from beard.utils.names import dm_tokenize_name, first_name_initial


class Precluster:

    """Representation of a precluster.

    Stores information about all the author names which belong to this
    precluster.
    """

    FULL_NAME_MATCH_BONUS = 10

    def __init__(self, tokens):
        """Create a precluster. Add first name.

        Parameters
        ----------
        :param tokens: tuple
            Tokens created from the author name in form of
            ((surname tokens), (first name tokens)).
        """
        self._content = {tokens[0]: {tokens[1]: 1}}

        # Note that in case where the cluster is created from a tuple of tokens
        # with multiple surnames, this signature will be counted to
        # _single_names_variants. This way we can omit dividing by 0.
        self._single_names_variants = 1
        self._name = copy(tokens[0][-1])

    def add_signature(self, tokens):
        """Add a signature to the precluster.

        Parameters
        ----------
        :param tokens: tuple
            Tokens created from the author name in form of
            ((surname tokens), (first name tokens)).
        """
        if tokens[0] in self._content:
            if tokens[1] in self._content[tokens[0]]:
                self._content[tokens[0]][tokens[1]] += 1
            else:
                self._content[tokens[0]][tokens[1]] = 1
        else:
            self._content[tokens[0]] = {tokens[1]: 1}

        if len(tokens[0]) == 1:
            self._single_names_variants += 1

    def compare_tokens_from_back(self, tokenized_prefix, last_name):
        """Match a part of the surname with given names in precluster.

        For example, ``Sanchez-Gomez, Juan`` can appear on a signature as
        ``Gomez, Juan Sanchez``. This function checks if there is a match
        between a signature like ``Sanchez-Gomez, Juan``, and names inside the
        precluster.

        Parameters
        ----------
        :param tokenized_prefix: tuple
            Tokens which represent all given names with few first surnames
            added. In form of a tuple of strings
        :param last_name: tuple
            Tokens, usually one, representing last surname(s) of the author.

        Returns
        -------
        :returns: boolean
            Information whether cluster contains this author if few last names
            are treated as first names.
        """
        if last_name in self._content:
            for first_names in six.iterkeys(self._content[last_name]):
                first_names_left = len(first_names)
                for reversed_index, name in \
                        enumerate(reversed(tokenized_prefix)):
                    if first_names_left == 0:
                        return True
                    elif first_names[-(reversed_index + 1)] != name:
                        break
                    first_names_left -= 1
                    if reversed_index == len(tokenized_prefix) - 1:
                        return True
            return False
        self._raise_keyerror(last_name)

    def contains(self, tokens):
        """Check if there is at least one signature with given surnames.

        Parameters
        ----------
        :param tokens: tuple
            Tokens which represent all surnames. Tuple of strings

        Returns
        -------
        :returns: boolean
            True if there is at least one sinature with given surnames.
        """
        return tokens in self._content

    def initials_score(self, new_first_names, last_name):
        """Count matches among the initials.

        All first names from the new signature should be matched. A match
        is full string match or a match between initials if at least one of the
        compared string is only an initial. An initial is the first character
        from the result of the double metaphone algorithm.

        Two characters in initials - ``A`` and ``H`` are handled in special
        way, as the double metaphone can output an empty string for them.

        Parameters
        ----------
        :param new_first_names: tuple
            Tuple of strings representing given names of the new author
        :param last_name: tuple
            Tokens, usually one, representing last surname(s) of the author.
        """
        result = 0
        if last_name in self._content:
            for first_names, occurences in \
                    six.iteritems(self._content[last_name]):
                first_names_length = len(first_names)
                old_names_index = 0
                names_match = self.FULL_NAME_MATCH_BONUS
                for initial in new_first_names:
                    while old_names_index < first_names_length:
                        full_names_match = \
                            self._same_initials(first_names[old_names_index],
                                                initial)
                        if not full_names_match:
                            old_names_index += 1
                        else:
                            if names_match == self.FULL_NAME_MATCH_BONUS:
                                names_match = full_names_match
                            break
                    if old_names_index == first_names_length:
                        names_match = False
                        break
                if names_match:
                    result += occurences * names_match
            return result
        self._raise_keyerror(last_name)

    def single_names_variants(self):
        """Return the number of signatures with only one surname.

        The only exception is that when the precluster was created by a
        signature which consisted of more than one surname, the result will
        be increased by one, so that the result can be used for normalization.


        Returns
        -------
        :returns: integer
            Number of signatures with only one surname. In case none are found,
            1.
        """
        return self._single_names_variants

    def _same_initials(self, name1, name2):

        if name1 == "" or name2 == "":
            # Probably starting with "h" or "w"
            if name1 != "":
                name1, name2 = name2, name1
            if name1 == name2:
                return True
            # Please not that names starting with "w" will result in strings
            # starting from "A" as the results of the double metaphone
            # algorithm.
            return name2.startswith('H') or name2.startswith('A')

        if len(name1) > 1 and len(name2) > 1:
            # Full names
            return self.FULL_NAME_MATCH_BONUS * (name1 == name2)

        # Just check initials
        return name1[0] == name2[0]

    def _raise_keyerror(self, key):
        raise KeyError("The cluster doesn't contain a key %s" % key)


def _split_blocks(blocks, X, threshold):

    splitted_blocks = []

    id_to_size = {}

    for block in blocks:
        if block._name in id_to_size:
            id_to_size[block._name] += 1
        else:
            id_to_size[block._name] = 1

    for index, precluster in enumerate(blocks):
        if id_to_size[precluster._name] > threshold:

            splitted_blocks.append(precluster._name +
                                   first_name_initial(X[index
                                                        ][0]['author_name']))
        else:
            splitted_blocks.append(precluster._name)

    return splitted_blocks


def dm_preclustering(X, threshold=1000):
    """Blah."""
    id_to_cluster = {}
    ordered_tokens = []

    for signature_array in X[:, 0]:
        tokens = dm_tokenize_name(signature_array['author_name'])
        surname_tokens = tokens[0]
        if len(surname_tokens) == 1:
            # Single surname case
            last_name = surname_tokens[0]
            if last_name not in id_to_cluster:
                id_to_cluster[last_name] = Precluster(tokens)
            else:
                id_to_cluster[last_name].add_signature(tokens)
            ordered_tokens.append((last_name,))
        else:
            ordered_tokens.append((None, tokens))

    blocks = []

    for token_tuple in ordered_tokens:
        if len(token_tuple) == 1:
            # There is already a block

            blocks.append(id_to_cluster[token_tuple[0]])
        else:
            # Case of multiple surnames

            tokens = token_tuple[1]
            last_metaphone_score = 0

            # Check if this combination of surnames was already included
            try:
                # First surname
                cluster = id_to_cluster[tokens[0][0]]
                if cluster.contains(tokens[0]):
                    cluster.add_signature(tokens)
                    blocks.append(cluster)
                    continue
            except KeyError:
                # No such block
                pass

            try:
                # Last surname
                cluster = id_to_cluster[tokens[0][-1]]
                if cluster.contains(tokens[0]):
                    cluster.add_signature(tokens)
                    blocks.append(cluster)
                    continue

                # No match, compute heuristically the match over initials

                index = len(tokens[0]) - 1
                # Here we need to consider every token prefix. For example
                # van der Somebody - (van, der) and (van) need to be
                # considered.

                match_found = False

                while index > 0:
                    token_prefix = tokens[0][:index]
                    if cluster.compare_tokens_from_back(token_prefix,
                                                        (tokens[0][-1],)):
                        cluster.add_signature(tokens)
                        match_found = True
                        break
                    index -= 1

                if match_found:
                    blocks.append(cluster)
                    continue

                # Second case is when the first name is dropped sometimes.
                # A good example might be a woman who took her husband's
                # surname as the first one.
                last_metaphone_score = \
                    cluster.initials_score(tokens[1], (tokens[0][-1],)) / \
                    float(cluster.single_names_variants())

            except KeyError:
                # No such block
                pass

            try:
                # First surname one more time
                cluster = id_to_cluster[tokens[0][0]]

                first_metaphone_score = 3 * \
                    cluster.initials_score(tokens[1], (tokens[0][0],)) / \
                    float(cluster.single_names_variants())

                if last_metaphone_score > first_metaphone_score:
                    id_to_cluster[tokens[0][-1]].add_signature(tokens)
                    blocks.append(id_to_cluster[tokens[0][-1]])
                else:
                    cluster.add_signature(tokens)
                    blocks.append(cluster)

                continue

            except KeyError:
                # No such block
                pass

            # No block for the first surname and no perfect match for the
            # last surname.
            if tokens[0][-1] not in id_to_cluster:
                id_to_cluster[tokens[0][-1]] = Precluster(tokens)
            blocks.append(id_to_cluster[tokens[0][-1]])

    return np.array(_split_blocks(blocks, X, threshold))
