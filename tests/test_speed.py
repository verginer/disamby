import pandas as pd
from faker import Faker
import disamby.preprocessors as prep
from disamby import Disamby
import pytest
import pandas as pd
import os


TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data')

@pytest.fixture
def pipeline():
    pipe = [prep.normalize_whitespace,
            prep.remove_punctuation,
            prep.compact_abbreviations,
            lambda x: prep.ngram(x, 4)]
    return pipe


@pytest.fixture
def company_df():
    assignee_path = os.path.join(DATA_DIR, 'potential_assginees_names.csv')
    return pd.read_csv(assignee_path, index_col='inv_id')


@pytest.mark.parametrize('size', [20, 1000, 2000])
def test_fitting(size, company_df, pipeline, benchmark):
    df = company_df.sample(size)
    benchmark(Disamby, df['name'], pipeline)


@pytest.mark.parametrize('size', [20, 1000, 2000, 8000])
def test_sparse_find(size, company_df, pipeline, benchmark):
    df = company_df.sample(size)
    dis = Disamby(df['name'], pipeline)
    term = list(dis._processed_token_cache['name'].keys())[0]
    results = benchmark(dis.find, term, 'name')
    score_of_searched = max(x.score for x in results)
    assert score_of_searched == pytest.approx(1)

