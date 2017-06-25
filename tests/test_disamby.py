#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from faker import Faker
import pandas as pd
from disamby import Disamby
import disamby.preprocessors as prep
from jellyfish import metaphone


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
                metaphone,
                prep.normalize_whitespace,
                prep.split_words]
    dis = Disamby()
    dis.fit(names, pipeline, 'streets')
    return dis


@pytest.fixture
def fake_pandas_df(fake_names):
    df = pd.DataFrame({
        'streets': fake_names(90, 40),
        'streets_2': fake_names(10, 40)
    })
    return df


def test_frequency_counter(disamby_fitted_instance):
    dis = disamby_fitted_instance
    assert 'streets' in dis.fields
    counter = dis.field_freq['streets']
    assert counter.most_common(1)[0][1] >= 1


def test_identification_potential(disamby_fitted_instance):
    dis = disamby_fitted_instance

    weights = dis.id_potential(('st', 'street', 'suite'), 'streets')
    assert sum(weights) == pytest.approx(1)


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

# test_scoring(smoother=None, offset=0, expected=.5454545454545454, disamby_fitted_instance=disamby_fitted_instance(fake_names()))


def test_dataframe(fake_pandas_df):
    df = fake_pandas_df
    test_idx = 20

    dis = Disamby()

    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                lambda x: prep.ngram(x, 4)]

    with pytest.raises(KeyError):
        dis.fit(df['streets'].values, pipeline)  # no field specified

    dis.fit(df['streets'], pipeline)

    with pytest.raises(ValueError):
        dis.score_df(test_idx, df)  # missing fitted field

    dis.fit(df['streets_2'], pipeline)
    scores = dis.score_df(test_idx, df)

    assert scores.loc[test_idx] == pytest.approx(1)  # score(a, a) === 1

    scores = dis.score_df(test_idx, df, smoother='log', offset=90)
    assert scores.loc[test_idx] == pytest.approx(1)

    scores = dis.score_df(test_idx, df, smoother='offset', offset=-90)
    assert scores.loc[test_idx] == pytest.approx(1)

    scores = dis.score_df(test_idx, df,
                          weight={'streets': .8, 'streets_2': .2},
                          smoother='log'
                          )
    assert scores.loc[test_idx] == pytest.approx(1)


def test_dataframe_as_argument(fake_pandas_df):
    df = fake_pandas_df
    test_idx = 20

    dis = Disamby()

    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                lambda x: prep.ngram(x, 4)]

    dis.fit(df, pipeline)

    scores = dis.score_df(test_idx, df)
    assert scores.loc[test_idx] == pytest.approx(1)  # score(a, a) === 1

    scores = dis.score_df(test_idx, df, smoother='log', offset=90)
    assert scores.max() == pytest.approx(1)

    scores = dis.score_df(test_idx, df, smoother='offset', offset=-90)
    assert scores.max() == pytest.approx(1)

    scores = dis.score_df(test_idx, df,
                          weight={'streets': .8, 'streets_2': .2},
                          smoother='log'
                          )
    assert scores.max() == pytest.approx(1)


def test_instant_instantiation(fake_pandas_df):
    df = fake_pandas_df
    test_idx = 20

    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                lambda x: prep.ngram(x, 4)]

    dis = Disamby(df, pipeline)

    scores = dis.score_df(test_idx, df, 'log')
    assert scores.max() == pytest.approx(1)


def test_log_scoring_pathological():
    from disamby.preprocessors import split_words, normalize_whitespace
    df = pd.DataFrame(
        {'a': ['Luca Georger', 'Luke Geroge', 'Adrian Sulzer'],
         'b': ['Mira, 34, Augsburg', 'Miri, 34, Augsburg', 'Milano, 34']}
    )
    prep = [normalize_whitespace, split_words]
    dis = Disamby(df, prep)
    score = dis.score_df(0, df, smoother='log', weight={'a': .2, 'b': .8})
    assert score[1] < 1
