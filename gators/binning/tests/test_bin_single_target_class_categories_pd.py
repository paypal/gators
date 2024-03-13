# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import BinSingleTargetClassCategories


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["_0", "_1", "_2", "_2", "_1"],
            "B": ["_1", "_2", "_1", "_1", "_1"],
            "C": ["_0", "_0", "_1", "_2", "_2"],
            "D": ["_0", "_0", "_1", "_1", "_1"],
            "E": [1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([0, 1, 1, 0, 0], name="Target")

    X_expected = pd.DataFrame(
        {
            "A": ["_0|_1", "_0|_1", "_2", "_2", "_0|_1"],
            "B": ["_1|_2", "_1|_2", "_1|_2", "_1|_2", "_1|_2"],
            "C": ["_0|_1|_2", "_0|_1|_2", "_0|_1|_2", "_0|_1|_2", "_0|_1|_2"],
            "D": ["_0", "_0", "_1", "_1", "_1"],
            "E": [1, 2, 3, 4, 5],
        }
    )
    obj = BinSingleTargetClassCategories().fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_binning():
    X = pd.DataFrame(
        {
            "A": ["_0", "_1", "_0", "_0", "_1"],
        }
    )
    y = pd.Series([0, 1, 1, 0, 0], name="Target")
    X_expected = pd.DataFrame(
        {
            "A": ["_0", "_1", "_0", "_0", "_1"],
        }
    )
    obj = BinSingleTargetClassCategories().fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_num():
    X = pd.DataFrame(
        {
            "D": [1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([0, 1, 1, 0, 0], name="Target")
    X_expected = pd.DataFrame(
        {
            "D": [1, 2, 3, 4, 5],
        }
    )
    obj = BinSingleTargetClassCategories().fit(X, y)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_binning_pd(data_no_binning):
    obj, X, X_expected = data_no_binning
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_binning_pd_np(data_no_binning):
    obj, X, X_expected = data_no_binning
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected.astype(object))


def test__num_pd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd__num_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected)
