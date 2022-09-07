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
def data_not_inplace():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF")),
        npartitions=1,
    )
    return MinMaxScaler(inplace=False).fit(X), X


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


def test_not_inplace_dd(data_not_inplace):
    obj, X = data_not_inplace
    X_new = obj.transform(X).compute()
    assert np.allclose(X_new.iloc[:, 5:].min().mean(), 0)
    assert np.allclose(X_new.iloc[:, 5:].max().mean(), 1)
    assert X_new.shape[1] == 10


def test_not_inplace_dd_np(data_not_inplace):
    obj, X = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    assert np.allclose(X_numpy_new[:, 5:].min().mean(), 0)
    assert np.allclose(X_numpy_new[:, 5:].max().mean(), 1)
    assert X_numpy_new.shape[1] == 10
