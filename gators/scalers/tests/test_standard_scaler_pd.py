# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest

from gators.scalers.standard_scaler import StandardScaler


@pytest.fixture
def data():
    X = pd.DataFrame(np.random.randn(5, 5), columns=list("ABCDF"))
    return StandardScaler().fit(X), X


@pytest.fixture
def data_not_inplace():
    X = pd.DataFrame(np.random.randn(5, 5), columns=list("ABCDF"))
    return StandardScaler(inplace=False).fit(X), X


def test_pd(data):
    obj, X = data
    X_new = obj.transform(X)
    assert np.allclose(X_new.mean().mean(), 0)
    assert np.allclose(X_new.std().mean(), 1)


def test_pd_np(data):
    obj, X = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new.mean().mean(), 0)
    assert np.allclose(X_new.std().mean(), 1)


def test_not_inplace_pd(data_not_inplace):
    obj, X = data_not_inplace
    X_new = obj.transform(X)
    assert np.allclose(X_new.iloc[:, 5:].mean(0).mean(), 0, atol=1e-7)
    assert np.allclose(X_new.iloc[:, 5:].std(0, ddof=1).mean(), 1, atol=1e-7)
    assert X_new.shape[1] == 10


def test_not_inplace_pd_np(data_not_inplace):
    obj, X = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    assert np.allclose(X_numpy_new[:, 5:].mean(0).mean(), 0, atol=1e-7)
    assert np.allclose(X_numpy_new[:, 5:].std(0, ddof=1).mean(), 1, atol=1e-7)
    assert X_numpy_new.shape[1] == 10
