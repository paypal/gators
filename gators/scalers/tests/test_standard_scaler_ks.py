# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from gators.scalers.standard_scaler import StandardScaler

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.random.randn(5, 5), columns=list("ABCDF"))
    return StandardScaler().fit(X), X


@pytest.fixture
def data_float32_ks():
    X = ks.DataFrame(np.random.randn(5, 5), columns=list("ABCDF"), dtype=np.float32)
    return StandardScaler().fit(X), X


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X = data_ks
    X_new = obj.transform(X)
    assert np.allclose(X_new.mean().mean(), 0)
    assert np.allclose(X_new.std().mean(), 1)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.mean().mean(), 0)
    assert np.allclose(X_new.std().mean(), 1)


@pytest.mark.koalas
def test_float32_ks(data_float32_ks):
    obj, X = data_float32_ks
    X_new = obj.transform(X)
    assert np.allclose(X_new.mean().mean(), 0, atol=1e-7)
    assert np.allclose(X_new.std().mean(), 1, atol=1e-7)


@pytest.mark.koalas
def test_float32_ks_np(data_float32_ks):
    obj, X = data_float32_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.mean().mean(), 0, atol=1e-7)
    assert np.allclose(X_new.std().mean(), 1, atol=1e-7)
