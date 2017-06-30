import disamby.preprocessors as prep
from disamby import Disamby
import pytest


pipeline = [prep.normalize_whitespace,
            prep.remove_punctuation,
            prep.compact_abbreviations,
            lambda x: prep.ngram(x, 4)]


@pytest.mark.parametrize('size', [20, 1000, 2000])
def test_fitting(size, company_df, benchmark):
    df = company_df(size)
    benchmark(Disamby, df, pipeline)


@pytest.mark.parametrize('size', [20, 1000, 2000])
def test_sparse_find(size, company_df, benchmark):
    df = company_df(size)
    dis = Disamby(df, pipeline)
    idx = list(dis.records['name'].keys())[0]
    results = benchmark(dis.find, idx, .7)
    score_of_searched = max(x.score for x in results)
    assert score_of_searched == pytest.approx(1)

