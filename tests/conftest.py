import os
import pytest
import pandas as pd
import numpy as np
from faker import Faker


TEST_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TEST_DIR, 'data')


@pytest.fixture
def fake_names():
    """function creating fake names of size n"""
    def f_names(n):
        fake = Faker()
        fake.seed(n)
        names = [fake.address() for _ in range(n)]
        return names
    return f_names


@pytest.fixture
def company_df(fake_names):
    """test company details"""
    def df_of_size(size):
        np.random.seed(size)
        fk_names = fake_names(size)
        assignee_path = os.path.join(DATA_DIR, 'potential_assginees_names.csv')
        sample = pd.read_csv(assignee_path, index_col='inv_id').sample(size)
        sample['address'] = fk_names
        del sample['patents']
        return sample
    return df_of_size
