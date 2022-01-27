# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders import BinnedColumnsEncoder


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["_0", "_0", "_1"],
                "B": ["_0", "_1", "_1"],
                "C": ["_1", "_0", "_1"],
                "D": [1.0, 2.0, 3.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["_0", "_0", "_1"],
            "B": ["_0", "_1", "_1"],
            "C": ["_1", "_0", "_1"],
            "D": [1.0, 2.0, 3.0],
            "A__ordinal": [0.0, 0.0, 1.0],
            "B__ordinal": [0.0, 1.0, 1.0],
            "C__ordinal": [1.0, 0.0, 1.0],
        }
    )
    obj = BinnedColumnsEncoder(columns=list("ABC"), inplace=False).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["_0", "_0", "_1"],
                "B": ["_0", "_1", "_1"],
                "C": ["_1", "_0", "_1"],
                "D": [1.0, 2.0, 3.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 1.0],
            "B": [0.0, 1.0, 1.0],
            "C": [1.0, 0.0, 1.0],
            "D": [1.0, 2.0, 3.0],
        }
    )
    obj = BinnedColumnsEncoder(columns=list("ABC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = dd.from_pandas(
        pd.DataFrame(
            np.zeros((3, 3)),
            columns=list("ABC"),
        ),
        npartitions=1,
    )
    obj = BinnedColumnsEncoder(columns=[]).fit(X)
    return obj, X, X.copy()


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_inplace_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_inplace_pd(data_inplace):
    obj, X, X_expected = data_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_pd_np(data_inplace):
    obj, X, X_expected = data_inplace
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_cat_pd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected.compute())


def test_no_cat_pd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.compute())
