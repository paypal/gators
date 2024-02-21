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
            "A__onehot__q": [1.0, 1.0, 0.0],
            "A__onehot__a": [0.0, 0.0, 1.0],
            "B__onehot__w": [1.0, 1.0, 0.0],
            "B__onehot__s": [0.0, 0.0, 1.0],
            "C__onehot__e": [1.0, 0.0, 0.0],
            "C__onehot__d": [0.0, 0.0, 1.0],
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
            "Aq": [1.0, 1.0, 0.0],
            "Aa": [0.0, 0.0, 1.0],
            "Bw": [1.0, 1.0, 0.0],
            "Bs": [0.0, 0.0, 1.0],
            "Ce": [1.0, 0.0, 0.0],
            "Cd": [0.0, 0.0, 1.0],
        }
    )
    column_names = ["Aq", "Aa", "Bw", "Bs", "Ce", "Cd"]
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict, column_names=column_names).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute()
    X_new[list("ABC")] = X_new[list("ABC")].astype(object)
    assert_frame_equal(X_new.iloc[:, -3:], X_expected.iloc[:, -3:])


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_dd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X).compute()
    X_new[list("ABC")] = X_new[list("ABC")].astype(object)
    assert_frame_equal(X_new, X_expected)


def test_dd_names_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
