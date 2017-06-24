#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from pytest import mark
from click.testing import CliRunner

from disamby import Disamby
from disamby.preprocessors import *
from disamby import cli


@mark.parametrize('raw,expected',[
    ('this is a   long  string', 'this is a long string'),
    ('this has a \tbut', 'this has a but',)
])
def test_reduce_duplicate_whitespace(raw, expected):
    assert reduce_duplicate_whitespace(raw) == expected


@mark.parametrize('raw,expected',[
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


def test_constructor():
    fields = ['location', 'name']
    D = Disamby(['location', 'name'])
    assert fields == D.fields
    assert set(fields) == D.field_freq.keys()


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'disamby.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
