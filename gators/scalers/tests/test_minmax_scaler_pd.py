# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest

from gators.scalers.minmax_scaler import MinMaxScaler


@pytest.fixture
def data():
    X = pd.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF"))
    return MinMaxScaler().fit(X), X


@pytest.fixture
def data_float32():
    X = pd.DataFrame(
        np.random.randn(5, 5),
        columns=list("ABCDF"),
    )
    return MinMaxScaler(dtype=np.float32).fit(X), X


def test_pd(data):
    obj, X = data
    X_new = obj.transform(X)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


def test_pd_np(data):
    obj, X = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


def test_float32_pd(data_float32):
    obj, X = data_float32
    X_new = obj.transform(X)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


def test_float32_pd_np(data_float32):
    obj, X = data_float32
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)
