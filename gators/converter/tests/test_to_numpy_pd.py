# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest

from gators.converter.to_numpy import ToNumpy


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "q": [0.0, 3.0, 6.0],
            "w": [1.0, 4.0, 7.0],
            "e": [2.0, 5.0, 8.0],
        }
    )
    y = pd.Series([0, 0, 1], name="TARGET")
    return X, y, X.to_numpy(), y.to_numpy()


def test_pd(data):
    X, y, X_expected, y_expected = data
    X_new, y_new = ToNumpy().transform(X, y)
    assert np.allclose(X_new, X_expected)
    assert np.allclose(y_new, y_expected)
