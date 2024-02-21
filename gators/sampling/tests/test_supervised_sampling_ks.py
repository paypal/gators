# License: Apache-2.0
import pyspark.pandas as ps
import numpy as np
import pytest

from gators.sampling import SupervisedSampling

ps.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    n_rows = 14
    n_cols = 5
    n_samples = 7
    X = ps.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    y = ps.Series(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            2,
            2,
        ],
        name="TARGET",
    )
    np.random.seed(1)
    obj = SupervisedSampling(frac_dict={0: 0.2, 1: 0.5, 2: 1.0})
    X_expected_shape = (5, 5)
    y_expected_shape = (5,)
    return obj, X, y, X_expected_shape, y_expected_shape


@pytest.mark.pyspark
def test_ks(data_ks):
    obj, X, y, X_expected_shape, y_expected_shape = data_ks
    X_new, y_new = obj.transform(X, y)
    assert X_new.shape == X_expected_shape
    assert y_new.shape == y_expected_shape
    assert (y_new == 0).sum() == 2
    assert (y_new == 1).sum() == 1
    assert (y_new == 2).sum() == 2
