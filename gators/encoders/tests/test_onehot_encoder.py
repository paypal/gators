# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.onehot_encoder import OneHotEncoder

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "D": {0: 1.0, 1: 2.0, 2: 3.0},
            "A__W": {0: 0.0, 1: 0.0, 2: 1.0},
            "A__Q": {0: 1.0, 1: 1.0, 2: 0.0},
            "B__W": {0: 0.0, 1: 1.0, 2: 1.0},
            "B__Q": {0: 1.0, 1: 0.0, 2: 0.0},
            "C__W": {0: 1.0, 1: 0.0, 2: 1.0},
            "C__Q": {0: 0.0, 1: 1.0, 2: 0.0},
        }
    )
    obj = OneHotEncoder().fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16():
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "D": {0: 1, 1: 2, 2: 3},
            "A__W": {0: 0, 1: 0, 2: 1},
            "A__Q": {0: 1, 1: 1, 2: 0},
            "B__W": {0: 0, 1: 1, 2: 1},
            "B__Q": {0: 1, 1: 0, 2: 0},
            "C__W": {0: 1, 1: 0, 2: 1},
            "C__Q": {0: 0, 1: 1, 2: 0},
        }
    ).astype(np.int16)
    obj = OneHotEncoder(dtype=np.int16).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = pd.DataFrame(
        np.arange(12).reshape(3, 4),
        columns=list("ABCD"),
        dtype=float,
    )
    obj = OneHotEncoder().fit(X)
    return obj, X, X.copy()


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
            "D": {0: 1.0, 1: 2.0, 2: 3.0},
            "A__W": {0: 0.0, 1: 0.0, 2: 1.0},
            "A__Q": {0: 1.0, 1: 1.0, 2: 0.0},
            "B__W": {0: 0.0, 1: 1.0, 2: 1.0},
            "B__Q": {0: 1.0, 1: 0.0, 2: 0.0},
            "C__W": {0: 1.0, 1: 0.0, 2: 1.0},
            "C__Q": {0: 0.0, 1: 1.0, 2: 0.0},
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
            "D": {0: 1, 1: 2, 2: 3},
            "A__W": {0: 0, 1: 0, 2: 1},
            "A__Q": {0: 1, 1: 1, 2: 0},
            "B__W": {0: 0, 1: 1, 2: 1},
            "B__Q": {0: 1, 1: 0, 2: 0},
            "C__W": {0: 1, 1: 0, 2: 1},
            "C__Q": {0: 0, 1: 1, 2: 0},
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


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_int16_pd(data_int16):
    obj, X, X_expected = data_int16
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_int16_ks(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_int16_pd_np(data_int16):
    obj, X, X_expected = data_int16
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_int16_ks_np(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_without_cat_pd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_without_cat_ks(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_without_cat_pd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_without_cat_ks_np(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
