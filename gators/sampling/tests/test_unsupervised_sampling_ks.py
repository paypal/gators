# License: Apache-2.0
import pyspark.pandas as ps
import numpy as np
import pandas as pd
import pytest

from gators.sampling import UnsupervisedSampling

ps.set_option("compute.default_index_type", "distributed-sequence")


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


@pytest.fixture
def data_ks():
    n_rows = 30
    n_cols = 5
    n_classes = 4
    n_samples = 5
    X = ps.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    np.random.seed(1)
    y = ps.Series(np.random.randint(0, n_classes, n_rows), name="TARGET")
    obj = UnsupervisedSampling(n_samples=n_samples)
    X_expected_shape = (5, 5)
    y_expected_shape = (5,)
    return obj, X, y, X_expected_shape, y_expected_shape


@pytest.mark.pyspark
def test_ks(data_ks):
    obj, X, y, X_expected_shape, y_expected_shape = data_ks
    X_new, y_new = obj.transform(X, y)
    assert X_new.shape == X_expected_shape
    assert y_new.shape == y_expected_shape
