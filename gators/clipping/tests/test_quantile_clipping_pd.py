# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.clipping import QuantileClipping


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [1.8, 2.2, 1.0, 0.4, 0.8],
            "B": [0.4, 1.9, -0.2, 0.1, 0.1],
            "C": [1.0, -1.0, -0.1, 1.5, 0.4],
        }
    )
    obj = QuantileClipping(
        columns=["A", "B", "C"], min_quantile=0.2, max_quantile=0.8
    ).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [1.8, 1.88, 1.0, 0.72, 0.8],
            "B": [0.4, 0.7, 0.04, 0.1, 0.1],
            "C": [1.0, -0.28, -0.1, 1.1, 0.4],
        }
    )
    return obj, X, X_expected


@pytest.fixture
def data_not_inplace():
    X = pd.DataFrame(
        {
            "A": [1.8, 2.2, 1.0, 0.4, 0.8],
            "B": [0.4, 1.9, -0.2, 0.1, 0.1],
            "C": [1.0, -1.0, -0.1, 1.5, 0.4],
        }
    )
    obj = QuantileClipping(
        columns=["A", "B", "C"], min_quantile=0.2, max_quantile=0.8, inplace=False
    ).fit(X)
    X_expected = pd.DataFrame(
        {
            "A__quantile_clip": [1.8, 1.88, 1.0, 0.72, 0.8],
            "B__quantile_clip": [0.4, 0.7, 0.04, 0.1, 0.1],
            "C__quantile_clip": [1.0, -0.28, -0.1, 1.1, 0.4],
        }
    )
    return obj, X, pd.concat([X, X_expected], axis=1)


@pytest.fixture
def data_partial():
    X = pd.DataFrame(
        {
            "A": [1.8, 2.2, 1.0, 0.4, 0.8],
            "B": [0.4, 1.9, -0.2, 0.1, 0.1],
            "C": [1.0, -1.0, -0.1, 1.5, 0.4],
        }
    )
    obj = QuantileClipping(min_quantile=0.2, max_quantile=0.8, columns=["A"]).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [1.8, 1.88, 1.0, 0.72, 0.8],
            "B": [0.4, 1.9, -0.2, 0.1, 0.1],
            "C": [1.0, -1.0, -0.1, 1.5, 0.4],
        }
    )
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


def test_not_inplace_pd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_not_inplace_pd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


def test_partial_pd(data_partial):
    obj, X, X_expected = data_partial
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_partial_pd_np(data_partial):
    obj, X, X_expected = data_partial
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


def test_init():
    with pytest.raises(TypeError):
        _ = QuantileClipping(columns=0, min_quantile=0.2, max_quantile=0.8)
    with pytest.raises(ValueError):
        _ = QuantileClipping(columns=[], min_quantile=0.2, max_quantile=0.8)
    with pytest.raises(TypeError):
        _ = QuantileClipping(columns=["A"], min_quantile="a", max_quantile=0.8)
    with pytest.raises(TypeError):
        _ = QuantileClipping(columns=["A"], min_quantile=0.2, max_quantile="a")
    with pytest.raises(TypeError):
        _ = QuantileClipping(
            columns=["A"], min_quantile=0.2, max_quantile=0.8, inplace="yes"
        )
