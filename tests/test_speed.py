import pandas as pd
from faker import Faker
import disamby.preprocessors as prep
from disamby import Disamby
import pytest


@pytest.fixture
def pipeline():
    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                lambda x: prep.ngram(x, 4)]
    return pipeline


def f_names(seed, n):
    fake = Faker()
    fake.seed(seed)
    names = [fake.address() for _ in range(n)]
    return names


@pytest.mark.parametrize('size', [20, 1000, 2000])
def test_instant_instantiation(size, pipeline, benchmark):
    df = pd.DataFrame({
        'streets': f_names(90, size),
        'streets_2': f_names(10, size)
    })

    dis = Disamby(df, pipeline)
    score_f = dis.pandas_score(0, df, 'log')
    scores = benchmark(df.apply, score_f, axis=1)
    assert scores.max() == pytest.approx(1)


def profile_function(size, pipeline):
    df = pd.DataFrame({
        'streets': f_names(90, size),
        'streets_2': f_names(10, size)
    })

    dis = Disamby(df, pipeline)
    score_f = dis.pandas_score(0, df, 'log')
    return df.apply(score_f, axis=1)


# profile_function(5000)
