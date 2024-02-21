# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.one_hot import OneHot


@pytest.fixture
def data():
    X = pd.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
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
    X = pd.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
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


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_pd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_names_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = OneHot(categories_dict=0)
    with pytest.raises(TypeError):
        _ = OneHot(categories_dict={"a": ["x"]}, column_names=0)
    with pytest.raises(ValueError):
        _ = OneHot(categories_dict={"a": ["x"]}, column_names=["aa", "bb"])
