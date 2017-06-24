#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from pytest import mark

from disamby.preprocessors import compact_abbreviations
from disamby.preprocessors import split_words
from disamby.preprocessors import reduce_duplicate_whitespace
from disamby.preprocessors import ngram


@mark.parametrize('raw,expected', [
    ('this is a   long  string', 'this is a long string'),
    ('this has a \tbut', 'this has a but',)
])
def test_reduce_duplicate_whitespace(raw, expected):
    assert reduce_duplicate_whitespace(raw) == expected


@mark.parametrize('raw,expected', [
    ('this is i.b.m', 'this is ibm'),
    ('an other A.B.M this', 'an other abm this')
])
def test_compact_abbreviations(raw, expected):
    assert compact_abbreviations(raw) == expected


@mark.parametrize('raw,n,expected', [
    ('this', 2, ('th', 'hi', 'is')),
    ('this', 3, ('thi', 'his')),
    ('this', 4, ('this',)),
    ('this', 5, ('this',)),
    ('this', 1, ValueError),
    ('this', -3, ValueError)
])
def test_ngram(raw, n, expected):
    if expected is ValueError:
        with pytest.raises(expected):
            ngram(raw, n)
    else:
        assert ngram(raw, n) == expected


def test_split_words():
    words = split_words("this is a sentence")
    assert len(words) == 4

    words = split_words("this")
    assert len(words) == 1


@mark.parametrize('raw,expected', [
    ('this Ias I.B.M', ('this', 'his ', 'is i', 's ia',
                        ' ias', 'ias ', 'as i', 's ib', ' ibm')
     )
])
def test_combined_preprocessors(raw, expected):
    reduced = reduce_duplicate_whitespace(raw)
    abbreviated = compact_abbreviations(reduced)
    trigram = ngram(abbreviated, 4)
    assert trigram == expected
