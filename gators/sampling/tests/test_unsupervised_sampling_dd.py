# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.sampling import UnsupervisedSampling


@pytest.fixture
def data():
    n_rows = 30
    n_cols = 5
    n_classes = 4
    n_samples = 5
    X = dd.from_pandas(
        pd.DataFrame(
            np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
        ),
        npartitions=1,
    )
    y = dd.from_pandas(
        pd.Series(np.random.randint(0, n_classes, n_rows), name="TARGET"), npartitions=1
    )
    obj = UnsupervisedSampling(n_samples=n_samples)
    X_expected_shape = (5, 5)
    y_expected_shape = (5,)
    return obj, X, y, X_expected_shape, y_expected_shape


def test_pd(data):
    obj, X, y, X_expected_shape, y_expected_shape = data
    X_new, y_new = obj.transform(X, y)
    assert X_new.compute().shape == X_expected_shape
    assert y_new.compute().shape == y_expected_shape


def test_init():
    with pytest.raises(TypeError):
        _ = UnsupervisedSampling(n_samples="a")


def test_no_sampling():
    X = dd.from_pandas(pd.DataFrame({"A": [0, 1, 2]}), npartitions=1)
    y = dd.from_pandas(pd.Series([0, 0, 1], name="TARGET"), npartitions=1)
    obj = UnsupervisedSampling(n_samples=5)
    assert_frame_equal(X.compute(), obj.transform(X, y)[0].compute())
    assert_series_equal(y.compute(), obj.transform(X, y)[1].compute())
