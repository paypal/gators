# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.target_encoder import TargetEncoder


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "Q", "W", "W", "W"],
                "B": ["Q", "Q", "W", "W", "W", "W"],
                "C": ["Q", "Q", "Q", "Q", "W", "W"],
                "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
        npartitions=2,
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=1)
    X_expected = pd.DataFrame(
        {
            "A": [
                0.0,
                0.0,
                0.0,
                0.6666666666666666,
                0.6666666666666666,
                0.6666666666666666,
            ],
            "B": [0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
            "C": [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    obj = TargetEncoder().fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = dd.from_pandas(
        pd.DataFrame(
            np.zeros((6, 3)),
            columns=list("ABC"),
        ),
        npartitions=2,
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=1)
    obj = TargetEncoder().fit(X, y)
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
