# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import QuantileBinning


@pytest.fixture
def data():
    n_bins = 4
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "C": ["a", "b", "c", "d", "e", "f"],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": ["_0", "_3", "_0", "_3", "_1", "_2"],
            "B__bin": ["_2", "_2", "_0", "_2", "_0", "_0"],
            "D__bin": ["_0", "_3", "_0", "_2", "_2", "_1"],
            "F__bin": ["_3", "_0", "_1", "_0", "_1", "_3"],
        }
    )
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16():
    n_bins = 4
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "C": ["a", "b", "c", "d", "e", "f"],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
            }
        ),
        npartitions=1,
    )
    X[list("ABDF")] = X[list("ABDF")].astype(np.int16)
    X_expected = (
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "C": ["a", "b", "c", "d", "e", "f"],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
                "A__bin": ["_0", "_3", "_0", "_3", "_1", "_2"],
                "B__bin": ["_2", "_2", "_0", "_2", "_0", "_0"],
                "D__bin": ["_0", "_3", "_0", "_2", "_2", "_1"],
                "F__bin": ["_3", "_0", "_1", "_0", "_1", "_3"],
            }
        ),
    )
    X_expected[list("ABDF")] = X_expected[list("ABDF")].astype(np.int16)
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace():
    n_bins = 4
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "C": ["a", "b", "c", "d", "e", "f"],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["_0", "_3", "_0", "_3", "_1", "_2"],
            "B": ["_2", "_2", "_0", "_2", "_0", "_0"],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": ["_0", "_3", "_0", "_2", "_2", "_1"],
            "F": ["_3", "_0", "_1", "_0", "_1", "_3"],
        }
    )
    obj = QuantileBinning(n_bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_num():
    X = dd.from_pandas(
        pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]}), npartitions=1
    )
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    n_bins = 3
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num():
    n_bins = 4
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": ["_0", "_3", "_0", "_3", "_1", "_2"],
            "B__bin": ["_2", "_2", "_0", "_2", "_0", "_0"],
            "D__bin": ["_0", "_3", "_0", "_2", "_2", "_1"],
            "F__bin": ["_3", "_0", "_1", "_0", "_1", "_3"],
        }
    )
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_inplace():
    n_bins = 4
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["_0", "_3", "_0", "_3", "_1", "_2"],
            "B": ["_2", "_2", "_0", "_2", "_0", "_0"],
            "D": ["_0", "_3", "_0", "_2", "_2", "_1"],
            "F": ["_3", "_0", "_1", "_0", "_1", "_3"],
        }
    )
    obj = QuantileBinning(n_bins, inplace=True).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_num_dd(data_no_num):
    obj, X, X_expected = data_no_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_no_num_dd_np(data_no_num):
    obj, X, X_expected = data_no_num
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_num_dd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_num_dd_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


# # inplace


def test_inplace_dd(data_inplace):
    obj, X, X_expected = data_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_inplace_dd_np(data_inplace):
    obj, X, X_expected = data_inplace
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_inplace_num_dd(data_num_inplace):
    obj, X, X_expected = data_num_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_inplace_num_dd_np(data_num_inplace):
    obj, X, X_expected = data_num_inplace
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))
