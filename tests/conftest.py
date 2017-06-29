import os
import pytest
import pandas as pd
import numpy as np

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data')


@pytest.fixture
def company_df():
    def df_of_size(size):
        np.random.seed(size)
        assignee_path = os.path.join(DATA_DIR, 'potential_assginees_names.csv')
        sample = pd.read_csv(assignee_path, index_col='inv_id').sample(size)
        return sample
    return df_of_size

