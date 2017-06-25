import pandas as pd
from faker import Faker
import disamby.preprocessors as prep
from disamby import Disamby
import pytest


def f_names(seed, n):
    fake = Faker()
    fake.seed(seed)
    names = [fake.address() for _ in range(n)]
    return names


def test_instant_instantiation(benchmark):
    df = pd.DataFrame({
        'streets': f_names(90, 1000),
        'streets_2': f_names(10, 1000)
    })

    pipeline = [prep.normalize_whitespace,
                prep.remove_punctuation,
                prep.compact_abbreviations,
                lambda x: prep.ngram(x, 4)]

    dis = Disamby(df, pipeline)

    scores = benchmark(dis.score_df, 90, df, 'log')
    assert scores.max() == pytest.approx(1)

# test_instant_instantiation()
