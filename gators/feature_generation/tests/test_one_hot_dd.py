# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.one_hot import OneHot


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC")),
        npartitions=1,
    )
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
def data_names():
    X = dd.from_pandas(
        pd.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC")),
        npartitions=1,
    )
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


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute().iloc[:, -3:], X_expected.iloc[:, -3:])


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_dd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_names_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
