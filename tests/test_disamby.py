#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from faker import Faker

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
    dis.fit('streets', names, pipeline)
    return dis


def test_frequency_counter(disamby_fitted_instance):
    dis = disamby_fitted_instance
    assert 'streets' in dis.fields
    counter = dis.field_freq['streets']
    assert counter.most_common(1)[0][1] >= 1


def test_identification_potential(disamby_fitted_instance):
    dis = disamby_fitted_instance

    weights = dis.id_potential(('st', 'street', 'suite'), 'streets')
    assert sum(weights) == pytest.approx(1)


def test_scoring(disamby_fitted_instance: Disamby):
    dis = disamby_fitted_instance
    score = dis.score('street george suit', 'suit street', 'streets')
    assert score == pytest.approx(.5454545454545454)

    score = dis.score('street george suit', 'suit street', 'streets',
                      smoother='offset', offset=1000)
    assert score > 0.66

    score = dis.score('street george suit', 'suit street', 'streets',
                      smoother='log', offset=10000)
    assert score <= .55

    with pytest.raises(KeyError):
        dis.score('street george suit', 'suit street', 'streets',
                          smoother='mambo', offset=10000)


def test_dataframe(fake_names):
    import pandas as pd

    test_idx = 20

    df = pd.DataFrame({
        'streets': fake_names(90, 40),
        'streets_2': fake_names(10, 40)
    })
    dis = Disamby()
    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                lambda x: prep.ngram(x, 4)]
    dis.fit('streets', df['streets'], pipeline)

    with pytest.raises(ValueError):
        # missing fitted field
        dis.score_df(test_idx, df)

    dis.fit('streets_2', df['streets_2'], pipeline)
    scores = dis.score_df(test_idx, df)
    # the score for the chosen individual must be 1 since score(a,a)=1
    assert scores.loc[test_idx] == pytest.approx(1)

    scores = dis.score_df(test_idx, df, smoother='log', offset=90)
    assert scores.loc[test_idx] == pytest.approx(1)

    scores = dis.score_df(test_idx, df, smoother='offset', offset=-90)
    assert scores.loc[test_idx] == pytest.approx(1)

    scores = dis.score_df(test_idx, df, weight={'streets': .8, 'streets_2': .2})
    assert scores.loc[test_idx] == pytest.approx(1)
