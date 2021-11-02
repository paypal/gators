# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.one_hot import OneHot

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
    X_expected = pd.DataFrame(
        {
            "A": ["q", "q", "a"],
            "B": ["w", "w", "s"],
            "C": ["e", "r", "d"],
            "A__onehot__q": [True, True, False],
            "A__onehot__a": [False, False, True],
            "B__onehot__w": [True, True, False],
            "B__onehot__s": [False, False, True],
            "C__onehot__e": [True, False, False],
            "C__onehot__d": [False, False, True],
        }
    )
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names_ks():
    X = ks.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
    X_expected = pd.DataFrame(
        {
            "A": ["q", "q", "a"],
            "B": ["w", "w", "s"],
            "C": ["e", "r", "d"],
            "Aq": [True, True, False],
            "Aa": [False, False, True],
            "Bw": [True, True, False],
            "Bs": [False, False, True],
            "Ce": [True, False, False],
            "Cd": [False, False, True],
        }
    )
    column_names = ["Aq", "Aa", "Bw", "Bs", "Ce", "Cd"]
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict, column_names=column_names).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas().iloc[:, -3:], X_expected.iloc[:, -3:])


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_names_np(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
