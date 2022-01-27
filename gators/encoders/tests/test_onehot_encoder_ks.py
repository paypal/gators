# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.onehot_encoder import OneHotEncoder

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "D": [1.0, 2.0, 3.0],
            "A__Q": [1.0, 1.0, 0.0],
            "A__W": [0.0, 0.0, 1.0],
            "B__Q": [1.0, 0.0, 0.0],
            "B__W": [0.0, 1.0, 1.0],
            "C__Q": [0.0, 1.0, 0.0],
            "C__W": [1.0, 0.0, 1.0],
        }
    )
    obj = OneHotEncoder().fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16_ks():
    X = ks.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "D": [1.0, 2.0, 3.0],
            "A__Q": [1.0, 1.0, 0.0],
            "A__W": [0.0, 0.0, 1.0],
            "B__Q": [1.0, 0.0, 0.0],
            "B__W": [0.0, 1.0, 1.0],
            "C__Q": [0.0, 1.0, 0.0],
            "C__W": [1.0, 0.0, 1.0],
        }
    ).astype(np.int16)
    obj = OneHotEncoder(dtype=np.int16).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat_ks():
    X = ks.DataFrame(
        np.arange(12).reshape(3, 4),
        columns=list("ABCD"),
        dtype=float,
    )
    obj = OneHotEncoder().fit(X)
    return obj, X, X.copy().to_pandas()


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_int16_ks(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_int16_ks_np(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_without_cat_ks(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_without_cat_ks_np(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
