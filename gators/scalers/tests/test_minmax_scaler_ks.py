# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from gators.scalers.minmax_scaler import MinMaxScaler

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF"))
    return MinMaxScaler().fit(X), X


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF"))
    return MinMaxScaler().fit(X), X


@pytest.fixture
def data_float32():
    X = pd.DataFrame(
        np.random.randn(5, 5),
        columns=list("ABCDF"),
    )
    return MinMaxScaler(dtype=np.float32).fit(X), X


@pytest.fixture
def data_float32_ks():
    X = pd.DataFrame(
        np.random.randn(5, 5),
        columns=list("ABCDF"),
    )
    return MinMaxScaler(dtype=np.float32).fit(X), X


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X = data_ks
    X_new = obj.transform(X)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


@pytest.mark.koalas
def test_float32_ks(data_float32_ks):
    obj, X = data_float32_ks
    X_new = obj.transform(X)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


@pytest.mark.koalas
def test_float32_ks_np(data_float32_ks):
    obj, X = data_float32_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)
