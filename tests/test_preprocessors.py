#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from pytest import mark

from disamby.preprocessors import compact_abbreviations
from disamby.preprocessors import split_words
from disamby.preprocessors import normalize_whitespace
from disamby.preprocessors import remove_punctuation
from disamby.preprocessors import ngram
from disamby.preprocessors import nword


@mark.parametrize('raw,expected', [
    ('this is a   long  string', 'THIS IS A LONG STRING'),
    ('this has a \tb', 'THIS HAS A B',)
])
def test_reduce_duplicate_whitespace(raw, expected):
    assert normalize_whitespace(raw) == expected


@mark.parametrize('raw,expected', [
    ('this is i.b.m', 'THIS IS IBM'),
    ('an other A.B.M this', 'AN OTHER ABM THIS')
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


def test_remove_punctuation():
    assert remove_punctuation('.has a .few!') == 'has a few'


@mark.parametrize('raw,expected', [
    ('this Ias I.B.M',
     ('THIS', 'HIS ', 'IS I', 'S IA', ' IAS', 'IAS ', 'AS I', 'S IB', ' IBM')
     )
])
def test_combined_preprocessors(raw, expected):
    reduced = normalize_whitespace(raw)
    abbreviated = compact_abbreviations(reduced)
    trigram = ngram(abbreviated, 4)
    assert trigram == expected


@pytest.mark.parametrize('raw,k,expected',[
    ('this that the other', 2, ('thisthat', 'thatthe', 'theother')),
    ('this that the other', 4, ('thisthattheother',)),
    ('this that the other', 20, ('thisthattheother',))
])
def test_word_sequencer(raw, k, expected):
    assert nword(raw, k) == expected
