# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import QuantileBinning


@pytest.fixture
def data():
    n_bins = 4
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": [
                "(-inf, 7.96)",
                "[41.83, inf)",
                "(-inf, 7.96)",
                "[41.83, inf)",
                "[7.96, 8.25)",
                "[8.25, 41.83)",
            ],
            "B__bin": [
                "[1.0, inf)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[0.0, 0.5)",
            ],
            "D__bin": [
                "(-inf, 26.3)",
                "[30.0, inf)",
                "(-inf, 26.3)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[26.3, 28.6)",
            ],
            "F__bin": [
                "[2.75, inf)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "[2.75, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace():
    n_bins = 4
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [
                "(-inf, 7.96)",
                "[41.83, inf)",
                "(-inf, 7.96)",
                "[41.83, inf)",
                "[7.96, 8.25)",
                "[8.25, 41.83)",
            ],
            "B": [
                "[1.0, inf)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[0.0, 0.5)",
            ],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [
                "(-inf, 26.3)",
                "[30.0, inf)",
                "(-inf, 26.3)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[26.3, 28.6)",
            ],
            "F": [
                "[2.75, inf)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "[2.75, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_num():
    X = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    n_bins = 3
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num():
    n_bins = 4
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": [
                "(-inf, 7.96)",
                "[41.83, inf)",
                "(-inf, 7.96)",
                "[41.83, inf)",
                "[7.96, 8.25)",
                "[8.25, 41.83)",
            ],
            "B__bin": [
                "[1.0, inf)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[0.0, 0.5)",
            ],
            "D__bin": [
                "(-inf, 26.3)",
                "[30.0, inf)",
                "(-inf, 26.3)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[26.3, 28.6)",
            ],
            "F__bin": [
                "[2.75, inf)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "[2.75, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins, inplace=False).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_inplace():
    n_bins = 4
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [
                "(-inf, 7.96)",
                "[41.83, inf)",
                "(-inf, 7.96)",
                "[41.83, inf)",
                "[7.96, 8.25)",
                "[8.25, 41.83)",
            ],
            "B": [
                "[1.0, inf)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[1.0, inf)",
                "[0.0, 0.5)",
                "[0.0, 0.5)",
            ],
            "D": [
                "(-inf, 26.3)",
                "[30.0, inf)",
                "(-inf, 26.3)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[26.3, 28.6)",
            ],
            "F": [
                "[2.75, inf)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "(-inf, 1.25)",
                "[2.0, 2.75)",
                "[2.75, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins, inplace=True).fit(X)
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


# # inplace


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
        _ = QuantileBinning(n_bins="a")
    with pytest.raises(TypeError):
        _ = QuantileBinning(n_bins=2, inplace="a")
