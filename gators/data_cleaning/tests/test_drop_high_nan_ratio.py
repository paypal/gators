import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_high_nan_ratio import DropHighNaNRatio


@pytest.fixture
def data():
    X = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan], "B": [np.nan, 0.0, 1.0], "C": ["a", "a", "b"]},
    )
    X_expected = pd.DataFrame(
        {"B": [np.nan, 0.0, 1.0], "C": ["a", "a", "b"]},
    )
    obj = DropHighNaNRatio(max_ratio=0.5).fit(X)
    return obj, obj, X, X, X_expected


@pytest.fixture
def data_no_drop():
    X = pd.DataFrame(
        {"A": list("qww"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = X.copy()
    obj = DropHighNaNRatio(max_ratio=0.5).fit(X)
    return obj, obj, X, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {"A": [np.nan, np.nan, np.nan], "B": [np.nan, 0.0, 1.0], "C": ["a", "a", "b"]},
    )
    X_expected = pd.DataFrame(
        {"B": [np.nan, 0.0, 1.0], "C": ["a", "a", "b"]},
    )
    obj = DropHighNaNRatio(max_ratio=0.5).fit(X)
    return obj, obj, X, X, X_expected


@pytest.fixture
def data_no_drop_ks():
    X = ks.DataFrame(
        {"A": list("qww"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = X.to_pandas().copy()
    obj = DropHighNaNRatio(max_ratio=0.5).fit(X)
    return obj, obj, X, X, X_expected


def test_pd(data):
    obj, obj, X, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    obj, obj, X, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_pd_np(data):
    obj, obj, X, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, obj, X, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_drop_pd(data_no_drop):
    obj, obj, X, X, X_expected = data_no_drop
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_drop_ks(data_no_drop_ks):
    obj, obj, X, X, X_expected = data_no_drop_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_no_drop_pd_np(data_no_drop):
    obj, obj, X, X, X_expected = data_no_drop
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_drop_ks_np(data_no_drop_ks):
    obj, obj, X, X, X_expected = data_no_drop_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = DropHighNaNRatio(max_ratio="q")
