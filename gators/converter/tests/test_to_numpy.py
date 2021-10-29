import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from gators.converter.to_numpy import ToNumpy

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "q": {0: 0.0, 1: 3.0, 2: 6.0},
            "w": {0: 1.0, 1: 4.0, 2: 7.0},
            "e": {0: 2.0, 1: 5.0, 2: 8.0},
        }
    )
    y = pd.Series([0, 0, 1], name="TARGET")
    return X, y, X.to_numpy(), y.to_numpy()


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "q": {0: 0.0, 1: 3.0, 2: 6.0},
            "w": {0: 1.0, 1: 4.0, 2: 7.0},
            "e": {0: 2.0, 1: 5.0, 2: 8.0},
        }
    )
    y = ks.Series([0, 0, 1], name="TARGET")
    return X, y, X.to_numpy(), y.to_numpy()


def test_pd(data):
    X, y, X_expected, y_expected = data
    X_new, y_new = ToNumpy().transform(X, y)
    assert np.allclose(X_new, X_expected)
    assert np.allclose(y_new, y_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    X, y, X_expected, y_expected = data_ks
    X_new, y_new = ToNumpy().transform(X, y)
    assert np.allclose(X_new, X_expected)
    assert np.allclose(y_new, y_expected)
