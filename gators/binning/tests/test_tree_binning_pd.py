# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from gators.binning import TreeBinning


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [1.07, -2.59, -1.54, 1.72],
            "B": [-1.19, -0.22, -0.28, 1.28],
            "C": [-1.15, 1.92, 1.09, -0.95],
            "D": ["a", "b", "c", "d"],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="TARGET")
    X_expected = pd.DataFrame(
        {
            "A": [1.07, -2.59, -1.54, 1.72],
            "B": [-1.19, -0.22, -0.28, 1.28],
            "C": [-1.15, 1.92, 1.09, -0.95],
            "D": ["a", "b", "c", "d"],
            "A__bin": ["_1", "_0", "_1", "_2"],
            "B__bin": ["_0", "_1", "_0", "_1"],
            "C__bin": ["_0", "_2", "_2", "_1"],
        }
    )
    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    obj = TreeBinning(tree=tree).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_regression():
    max_depth = 2
    X = pd.DataFrame(
        {
            "A": [-0.1, 1.45, 0.98, -0.98],
            "B": [-0.15, 0.14, 0.4, 1.87],
            "C": [0.95, 0.41, 1.76, 2.24],
        }
    )
    y = pd.Series([39.9596835, 36.65644911, 137.24445075, 300.15325913], name="TARGET")
    X_expected = pd.DataFrame(
        {
            "A": ["_1", "_2", "_1", "_0"],
            "B": ["_0", "_0", "_1", "_2"],
            "C": ["_0", "_0", "_1", "_2"],
        }
    )
    tree = DecisionTreeRegressor(max_depth=2, random_state=0)
    obj = TreeBinning(tree=tree, inplace=True).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_num():
    X = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    y = pd.Series([0, 1, 2, 3, 4, 5], name="TARGET")
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    obj = TreeBinning(tree=tree).fit(X, y)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_regression_pd(data_regression):
    obj, X, X_expected = data_regression
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_regression_pd_np(data_regression):
    obj, X, X_expected = data_regression
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_num_pd(data_no_num):
    obj, X, X_expected = data_no_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_num_pd_np(data_no_num):
    obj, X, X_expected = data_no_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = TreeBinning(max_depth="a")
