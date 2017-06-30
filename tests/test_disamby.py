#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from faker import Faker
import pandas as pd
from disamby import Disamby
import disamby.preprocessors as prep


@pytest.fixture
def fake_names():
    def f_names(seed, n):
        fake = Faker()
        fake.seed(seed)
        names = [fake.address() for _ in range(n)]
        return names
    return f_names


@pytest.fixture
def disamby_fitted_instance(fake_names):
    names = fake_names(90, 20)
    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                prep.split_words]
    data_series = pd.Series(names)
    dis = Disamby()
    dis.fit(data_series, pipeline, 'streets')
    return dis


@pytest.fixture
def fake_pandas_df(fake_names):
    df = pd.DataFrame({
        'streets': fake_names(90, 40),
        'streets_2': fake_names(10, 40)
    })
    return df


pipeline = [
    prep.normalize_whitespace,
    prep.remove_punctuation,
    prep.compact_abbreviations,
    lambda x: prep.ngram(x, 4)
]


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
    (None, 0, 1),
    ('offset', 1000, 1),
    ('log', 10000, 1),
])
def test_scoring(smoother, offset, expected, disamby_fitted_instance):
    dis = disamby_fitted_instance
    score = dis.score('David Heights', 'Rebecca Shoal Suite', 'streets',
                      smoother=smoother, offset=offset)
    assert score == pytest.approx(expected, abs=.01)


def test_dataframe_raise_exception(fake_pandas_df):
    df = fake_pandas_df
    dis = Disamby()

    with pytest.raises(KeyError):
        dis.fit(df['streets'].values, pipeline)  # no field specified

    with pytest.raises(ValueError):
        dis.pandas_score(20, df)  # missing fitted field


@pytest.mark.parametrize('smoother,offset,weight,expected',[
    (None, 0, None, 0.011928429423459246),
    ('log', 90, None, 0.02741944357386672),
    ('offset', -90, None, 0.03125),
    ('log', 90, {'streets': .8, 'streets_2': .2}, 0.010967777429546688)
])
def test_dataframe(smoother, offset, weight, expected, fake_pandas_df):
    df = fake_pandas_df
    test_idx = 20

    dis = Disamby(df, pipeline)

    scoring_fun = dis.pandas_score(test_idx, df, smoother, offset, weight)
    scores = df.apply(scoring_fun, axis=1)

    assert scores.loc[test_idx] == pytest.approx(1)  # score(a, a) === 1
    assert ((scores.values >= 0) & (scores.values <= 1.001)).all()
    assert scores.loc[2] == pytest.approx(expected)


def test_dataframe_as_argument(fake_pandas_df):
    df = fake_pandas_df
    test_idx = 20

    dis = Disamby()
    dis.fit(df, pipeline)

    score_f = dis.pandas_score(test_idx, df)
    scores = df.apply(score_f, axis=1)
    assert scores.loc[test_idx] == pytest.approx(1)  # score(a, a) === 1


def test_instant_instantiation(fake_pandas_df):
    df = fake_pandas_df
    dis = Disamby(df, pipeline)
    dis.field_freq['streets'].most_common()


def test_log_scoring_pathological():
    df = pd.DataFrame(
        {'a': ['Luca Georger', 'Luke Geroge', 'Adrian Sulzer'],
         'b': ['Mira, 34, Augsburg', 'Miri, 34, Augsburg', 'Milano, 34']}
    )
    dis = Disamby(df, pipeline)
    score_f = dis.pandas_score(0, df, smoother='log', weight={'a': .2, 'b': .8})
    score = df.apply(score_f, axis=1)
    assert score[1] < 1


def test_find(fake_pandas_df):
    df = fake_pandas_df
    dis = Disamby(df, pipeline)
    _, term = dis.records['streets'][0]
    results = dis.find(term, 'streets')

    assert len(results) == 5
    score_of_searched = max(x.score for x in results)
    assert score_of_searched == pytest.approx(1)


def test_alias_graph(company_df):
    from networkx import strongly_connected_components
    df = company_df(200)
    dis = Disamby(df['name'], preprocessors=pipeline)
    graph = dis.alias_graph('name', verbose=True)
    components = strongly_connected_components(graph)
    assert max(len(c) for c in components) == 2
