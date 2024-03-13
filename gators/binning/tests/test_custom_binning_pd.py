# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import CustomBinning


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": [
                "(-inf, 8.0)",
                "[40.0, inf)",
                "(-inf, 8.0)",
                "[40.0, inf)",
                "[8.0, 40.0)",
                "[8.0, 40.0)",
            ],
            "D__bin": [
                "(-inf, 30.0)",
                "[30.0, inf)",
                "(-inf, 30.0)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[30.0, inf)",
            ],
            "F__bin": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 2.0, np.inf],
    }
    obj = CustomBinning(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_num():
    X = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 2.0, np.inf],
    }
    obj = CustomBinning(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [
                "(-inf, 8.0)",
                "[40.0, inf)",
                "(-inf, 8.0)",
                "[40.0, inf)",
                "[8.0, 40.0)",
                "[8.0, 40.0)",
            ],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [
                "(-inf, 30.0)",
                "[30.0, inf)",
                "(-inf, 30.0)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[30.0, inf)",
            ],
            "F": [
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
            ],
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, np.inf],
    }
    obj = CustomBinning(bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": [
                "(-inf, 8.0)",
                "[40.0, inf)",
                "(-inf, 8.0)",
                "[40.0, inf)",
                "[8.0, 40.0)",
                "[8.0, 40.0)",
            ],
            "D__bin": [
                "(-inf, 30.0)",
                "[30.0, inf)",
                "(-inf, 30.0)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[30.0, inf)",
            ],
            "F__bin": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 2.0, np.inf],
    }
    obj = CustomBinning(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_inplace():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [
                "(-inf, 8.0)",
                "[40.0, inf)",
                "(-inf, 8.0)",
                "[40.0, inf)",
                "[8.0, 40.0)",
                "[8.0, 40.0)",
            ],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [
                "(-inf, 30.0)",
                "[30.0, inf)",
                "(-inf, 30.0)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[30.0, inf)",
            ],
            "F": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 2, np.inf],
    }
    obj = CustomBinning(bins, inplace=True).fit(X)
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


def test_num_pd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_num_pd_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


# # # inplace


def test_inplace_pd(data_inplace):
    obj, X, X_expected = data_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_inplace_pd_np(data_inplace):
    obj, X, X_expected = data_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_inplace_num_pd(data_num_inplace):
    obj, X, X_expected = data_num_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_inplace_num_pd_np(data_num_inplace):
    obj, X, X_expected = data_num_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = CustomBinning(bins_dict="a")
    with pytest.raises(TypeError):
        _ = CustomBinning(bins_dict={"A": [-np.inf, np.inf]}, inplace="a")
