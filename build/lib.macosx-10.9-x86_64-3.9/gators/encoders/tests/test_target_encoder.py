# License: Apache-2.0
from gators.encoders.target_encoder import TargetEncoder
from pandas.testing import assert_frame_equal
import pytest
import numpy as np
import pandas as pd
import databricks.koalas as ks
ks.set_option('compute.default_index_type', 'distributed-sequence')


@pytest.fixture
def data():
    X = pd.DataFrame({
            'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
            'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
            'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
            'D': [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 0, 1, 1, 0], name='TARGET')
    X_expected = pd.DataFrame({
            'A': {0: 0.0,
                  1: 0.0,
                  2: 0.0,
                  3: 0.6666666666666666,
                  4: 0.6666666666666666,
                  5: 0.6666666666666666},
            'B': {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            'C': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.5, 5: 0.5},
            'D': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}})
    obj = TargetEncoder().fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_float32():
    X = pd.DataFrame({
            'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
            'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
            'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
            'D': [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 0, 1, 1, 0], name='TARGET')
    X_expected = pd.DataFrame({
            'A': {0: 0.0,
                  1: 0.0,
                  2: 0.0,
                  3: 0.6666666666666666,
                  4: 0.6666666666666666,
                  5: 0.6666666666666666},
            'B': {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            'C': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.5, 5: 0.5},
            'D': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}}).astype(np.float32)
    obj = TargetEncoder(dtype=np.float32).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = pd.DataFrame(
        np.zeros((6, 3)),
        columns=list('ABC'),
    )
    y = pd.Series([0, 0, 0, 1, 1, 0], name='TARGET')
    obj = TargetEncoder().fit(X, y)
    return obj, X, X.copy()


@pytest.fixture
def data_ks():
    X = ks.DataFrame({
            'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
            'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
            'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
            'D': [1, 2, 3, 4, 5, 6]})
    y = ks.Series([0, 0, 0, 1, 1, 0], name='TARGET')
    X_expected = pd.DataFrame({
            'A': {0: 0.0,
                  1: 0.0,
                  2: 0.0,
                  3: 0.6666666666666666,
                  4: 0.6666666666666666,
                  5: 0.6666666666666666},
            'B': {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            'C': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.5, 5: 0.5},
            'D': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}})
    obj = TargetEncoder().fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_float32_ks():
    X = ks.DataFrame({
            'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
            'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
            'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
            'D': [1, 2, 3, 4, 5, 6]})
    y = ks.Series([0, 0, 0, 1, 1, 0], name='TARGET')
    X_expected = pd.DataFrame({
            'A': {0: 0.0,
                  1: 0.0,
                  2: 0.0,
                  3: 0.6666666666666666,
                  4: 0.6666666666666666,
                  5: 0.6666666666666666},
            'B': {0: 0.0, 1: 0.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},
            'C': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.5, 5: 0.5},
            'D': {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}}).astype(np.float32)
    obj = TargetEncoder(dtype=np.float32).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat_ks():
    X = ks.DataFrame(
        np.zeros((6, 3)),
        columns=list('ABC'),
    )
    y = ks.Series([0, 0, 0, 1, 1, 0], name='TARGET')
    obj = TargetEncoder().fit(X, y)
    return obj, X, X.to_pandas().copy()


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


def test_float32_pd(data_float32):
    obj, X, X_expected = data_float32
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_float32_ks(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_float32_pd_np(data_float32):
    obj, X, X_expected = data_float32
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_float32_ks_np(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_pd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_cat_ks(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_no_cat_pd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_cat_ks_np(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
