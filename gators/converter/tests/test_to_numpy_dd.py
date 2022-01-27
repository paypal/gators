# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from gators.converter.to_numpy import ToNumpy


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "q": [0.0, 3.0, 6.0],
                "w": [1.0, 4.0, 7.0],
                "e": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    y = dd.from_pandas(pd.Series([0, 0, 1], name="TARGET"), npartitions=1)
    return X, y, X.compute().to_numpy(), y.compute().to_numpy()


def test_pd(data):
    X, y, X_expected, y_expected = data
    X_new, y_new = ToNumpy().transform(X, y)
    assert np.allclose(X_new, X_expected)
    assert np.allclose(y_new, y_expected)
