#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `disamby` package."""
import pytest
from faker import Faker

from disamby import Disamby
from disamby.preprocessors import *


def fake_names():
    fake = Faker()
    fake.seed(90)
    names = [fake.address() for _ in range(20)]
    return names


@pytest.fixture
def disamby_fitted_instance():
    names = fake_names()
    pipeline = [reduce_duplicate_whitespace, compact_abbreviations, split_words]
    dis = Disamby()
    dis.fit('streets', names, pipeline)
    return dis


def test_frequency_counter(disamby_fitted_instance):
    dis = disamby_fitted_instance
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
