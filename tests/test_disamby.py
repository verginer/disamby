#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from faker import Faker

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
    pipeline = [prep.reduce_duplicate_whitespace,
                prep.compact_abbreviations,
                prep.split_words]
    dis = Disamby()
    dis.fit('streets', names, pipeline)
    return dis


def test_frequency_counter(disamby_fitted_instance):
    dis = disamby_fitted_instance
    assert 'streets' in dis.fields
    counter = dis.field_freq['streets']
    assert counter.most_common(1) == [('unit', 5)]


def test_identification_potential(disamby_fitted_instance):
    dis = disamby_fitted_instance

    weights = dis.identification_weight(('st', 'street', 'suite'), 'streets')
    assert sum(weights) == pytest.approx(1)


def test_scoring(disamby_fitted_instance: Disamby):
    dis = disamby_fitted_instance
    score = dis.score('street george suit', 'suit street', 'streets')
    assert score == pytest.approx(0.6)


def test_dataframe(fake_names):
    import pandas as pd
    df = pd.DataFrame({'streets': fake_names(90, 40),
                       'streets_2': fake_names(10, 40)
                       })
    dis = Disamby()
    pipeline = [prep.reduce_duplicate_whitespace,
                prep.compact_abbreviations,
                prep.split_words]
    dis.fit('streets', df['streets'], pipeline)
    dis.fit('streets_2', df['streets_2'], pipeline)

    test_guy = 20
    scores = dis.score_df(test_guy, df)
    assert scores.loc[test_guy] == pytest.approx(1)
