# License: Apache-2.0
import numpy as np
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders import BinnedColumnsEncoder

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = ks.DataFrame(
        {
            "A": ["_0", "_0", "_1"],
            "B": ["_0", "_1", "_1"],
            "C": ["_1", "_0", "_1"],
            "D": [1.0, 2.0, 3.0],
        }
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
    X = ks.DataFrame(
        {
            "A": ["_0", "_0", "_1"],
            "B": ["_0", "_1", "_1"],
            "C": ["_1", "_0", "_1"],
            "D": [1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 1.0],
            "B": [0.0, 1.0, 1.0],
            "C": [1.0, 0.0, 1.0],
            "D": [1, 2, 3],
        }
    )
    obj = BinnedColumnsEncoder(columns=list("ABC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = ks.DataFrame(
        np.zeros((3, 3)),
        columns=list("ABC"),
    )
    obj = BinnedColumnsEncoder(columns=[]).fit(X)
    return obj, X, X.copy()


@pytest.mark.koalas
def test_ks(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_inplace_ks(data_inplace):
    obj, X, X_expected = data_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_inplace_ks_np(data_inplace):
    obj, X, X_expected = data_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_cat_ks(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected.to_pandas())


@pytest.mark.koalas
def test_no_cat_ks_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.to_pandas())
