#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
import pandas as pd
from disamby import Disamby
import disamby.preprocessors as prep


pipeline = [
    prep.normalize_whitespace,
    prep.remove_punctuation,
    prep.compact_abbreviations,
    lambda x: prep.ngram(x, 4)
]


@pytest.fixture
def disamby_fitted_instance(fake_names):
    names = fake_names(90)
    data_series = pd.Series(names)
    dis = Disamby()
    dis.fit(data_series, pipeline, 'streets')
    return dis



def test_frequency_counter(disamby_fitted_instance):
    dis = disamby_fitted_instance
    assert 'streets' in dis.fields
    counter = dis.field_freq['streets']
    assert counter.most_common(1)[0][1] >= 1


def test_identification_potential(disamby_fitted_instance):
    dis = disamby_fitted_instance

    weights = dis.id_potential(('st', 'street', 'suite'), 'streets')
    assert sum(weights.values()) == pytest.approx(1)


@pytest.mark.parametrize('smoother,offset,expected', [
    (None, 0, 0),
    ('offset', 1000, 0),
    ('log', 10000, 0),
])
def test_scoring(smoother, offset, expected, disamby_fitted_instance):
    dis = disamby_fitted_instance
    score = dis.score('David Heights', 'Rebecca Shoal Suite', 'streets',
                      smoother=smoother, offset=offset)
    assert score == pytest.approx(expected, abs=.01)


def test_instant_instantiation(company_df):
    df = company_df(500)
    dis = Disamby(df, pipeline)
    dis.field_freq['address'].most_common()


def test_find(company_df):
    df = company_df(100)
    dis = Disamby(df, pipeline)
    term = list(dis.records['address'].keys())[0]
    results = dis.find(term, threshold=0.001, weights={'name': .2, 'address': .8})

    assert len(results) == 42
    score_of_searched = max(x.score for x in results)
    assert score_of_searched == pytest.approx(1)


def test_alias_graph(company_df):
    from networkx import strongly_connected_components
    df = company_df(200)
    dis = Disamby(df, preprocessors=pipeline)
    graph = dis.alias_graph(verbose=True, threshold=0.7,
                            weights={'name': .99, 'address': .01}
                            )
    components = strongly_connected_components(graph)
    assert max(len(c) for c in components) == 2
