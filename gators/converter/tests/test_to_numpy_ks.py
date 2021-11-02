# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from gators.converter.to_numpy import ToNumpy

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "q": [0.0, 3.0, 6.0],
            "w": [1.0, 4.0, 7.0],
            "e": [2.0, 5.0, 8.0],
        }
    )
    y = ks.Series([0, 0, 1], name="TARGET")
    return X, y, X.to_numpy(), y.to_numpy()


@pytest.mark.koalas
def test_ks(data_ks):
    X, y, X_expected, y_expected = data_ks
    X_new, y_new = ToNumpy().transform(X, y)
    assert np.allclose(X_new, X_expected)
    assert np.allclose(y_new, y_expected)
