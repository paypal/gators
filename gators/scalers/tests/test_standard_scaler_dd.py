# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from gators.scalers.standard_scaler import StandardScaler


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(np.random.randn(5, 5), columns=list("ABCDF")), npartitions=1
    )
    return StandardScaler().fit(X), X


@pytest.fixture
def data_float32():
    X = dd.from_pandas(
        pd.DataFrame(np.random.randn(5, 5), columns=list("ABCDF"), dtype=np.float32),
        npartitions=1,
    )
    return StandardScaler().fit(X), X


def test_pd(data):
    obj, X = data
    X_new = obj.transform(X)
    assert np.allclose(X_new.compute().mean().mean(), 0)
    assert np.allclose(X_new.compute().std().mean(), 1)


def test_pd_np(data):
    obj, X = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.mean().mean(), 0)
    assert np.allclose(X_new.std().mean(), 1)


def test_float32_pd(data_float32):
    obj, X = data_float32
    X_new = obj.transform(X)
    assert np.allclose(X_new.compute().mean().mean(), 0, atol=1e-7)
    assert np.allclose(X_new.compute().std().mean(), 1, atol=1e-7)


def test_float32_pd_np(data_float32):
    obj, X = data_float32
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.mean().mean(), 0, atol=1e-7)
    assert np.allclose(X_new.std().mean(), 1, atol=1e-7)
