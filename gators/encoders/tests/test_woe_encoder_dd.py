# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.woe_encoder import WOEEncoder


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "Q", "W", "W", "W"],
                "B": ["Q", "Q", "W", "W", "W", "W"],
                "C": ["Q", "Q", "Q", "Q", "W", "W"],
                "D": [1, 2, 3, 4, 5, 6],
            }
        ),
        npartitions=2,
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=1)
    X_expected = pd.DataFrame(
        {
            "A": [
                -1.9459101490553135,
                -1.9459101490553135,
                -1.9459101490553135,
                0.5108256237659907,
                0.5108256237659907,
                0.5108256237659907,
            ],
            "B": [-1.6094379124341003, -1.6094379124341003, 0.0, 0.0, 0.0, 0.0],
            "C": [
                -0.8472978603872037,
                -0.8472978603872037,
                -0.8472978603872037,
                -0.8472978603872037,
                0.0,
                0.0,
            ],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    obj = WOEEncoder(regularization=0.5).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_float32():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "Q", "W", "W", "W"],
                "B": ["Q", "Q", "W", "W", "W", "W"],
                "C": ["Q", "Q", "Q", "Q", "W", "W"],
                "D": [1, 2, 3, 4, 5, 6],
            }
        ),
        npartitions=2,
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=1)
    X_expected = pd.DataFrame(
        {
            "A": [
                -1.9459101490553135,
                -1.9459101490553135,
                -1.9459101490553135,
                0.5108256237659907,
                0.5108256237659907,
                0.5108256237659907,
            ],
            "B": [-1.6094379124341003, -1.6094379124341003, 0.0, 0.0, 0.0, 0.0],
            "C": [
                -0.8472978603872037,
                -0.8472978603872037,
                -0.8472978603872037,
                -0.8472978603872037,
                0.0,
                0.0,
            ],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).astype(np.float32)
    obj = WOEEncoder(regularization=0.5, dtype=np.float32).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = dd.from_pandas(
        pd.DataFrame(np.zeros((6, 3)), columns=list("ABC")), npartitions=2
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=1)
    obj = WOEEncoder().fit(X, y)
    return obj, X, X.compute().copy()


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_float32_dd(data_float32):
    obj, X, X_expected = data_float32
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_float32_dd_np(data_float32):
    obj, X, X_expected = data_float32
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_dd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_no_cat_dd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
