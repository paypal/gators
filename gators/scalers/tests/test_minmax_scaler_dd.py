# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from gators.scalers.minmax_scaler import MinMaxScaler


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF")),
        npartitions=1,
    )
    return MinMaxScaler().fit(X), X


@pytest.fixture
def data_float32():
    X = dd.from_pandas(
        pd.DataFrame(
            np.random.randn(5, 5),
            columns=list("ABCDF"),
        ),
        npartitions=1,
    )
    return MinMaxScaler(dtype=np.float32).fit(X), X


def test_dd(data):
    obj, X = data
    X_new = obj.transform(X)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


def test_dd_np(data):
    obj, X = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


def test_float32_dd(data_float32):
    obj, X = data_float32
    X_new = obj.transform(X)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)


def test_float32_dd_np(data_float32):
    obj, X = data_float32
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.min().mean(), 0)
    assert np.allclose(X_new.max().mean(), 1)
