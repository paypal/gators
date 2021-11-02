# License: Apache-2.0
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
    X = pd.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    y = pd.Series(np.random.randint(0, n_classes, n_rows), name="TARGET")
    obj = UnsupervisedSampling(n_samples=n_samples)
    X_expected_shape = (5, 5)
    y_expected_shape = (5,)
    return obj, X, y, X_expected_shape, y_expected_shape


def test_pd(data):
    obj, X, y, X_expected_shape, y_expected_shape = data
    X_new, y_new = obj.transform(X, y)
    assert X_new.shape == X_expected_shape
    assert y_new.shape == y_expected_shape


def test_init():
    with pytest.raises(TypeError):
        _ = UnsupervisedSampling(n_samples="a")


def test_no_sampling():
    X = pd.DataFrame({"A": [0, 1, 2]})
    y = pd.Series([0, 0, 1], name="TARGET")
    obj = UnsupervisedSampling(n_samples=5)
    assert_frame_equal(X, obj.transform(X, y)[0])
    assert_series_equal(y, obj.transform(X, y)[1])
