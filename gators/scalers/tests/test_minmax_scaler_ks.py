# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from gators.scalers.minmax_scaler import MinMaxScaler

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF"))
    return MinMaxScaler().fit(X), X


@pytest.fixture
def data_not_inplace():
    X = ks.DataFrame(np.arange(25).reshape((5, 5)), columns=list("ABCDF"))
    return MinMaxScaler(inplace=False).fit(X), X


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
def test_not_inplace_pd(data_not_inplace):
    obj, X = data_not_inplace
    X_new = obj.transform(X).to_pandas()
    assert np.allclose(X_new.iloc[:, 5:].min().mean(), 0)
    assert np.allclose(X_new.iloc[:, 5:].max().mean(), 1)
    assert X_new.shape[1] == 10


@pytest.mark.koalas
def test_not_inplace_pd_np(data_not_inplace):
    obj, X = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    assert np.allclose(X_numpy_new[:, 5:].min().mean(), 0)
    assert np.allclose(X_numpy_new[:, 5:].max().mean(), 1)
    assert X_numpy_new.shape[1] == 10
