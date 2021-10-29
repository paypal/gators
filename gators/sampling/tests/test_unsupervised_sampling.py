# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.sampling import UnsupervisedSampling

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    n_rows = 30
    n_cols = 5
    n_classes = 4
    n_samples = 5
    X = pd.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    np.random.seed(1)
    y = pd.Series(np.random.randint(0, n_classes, n_rows), name="TARGET")
    obj = UnsupervisedSampling(n_samples=n_samples)
    X_expected = pd.DataFrame(
        {
            "A": {2: 10, 28: 140, 13: 65, 10: 50, 26: 130},
            "B": {2: 11, 28: 141, 13: 66, 10: 51, 26: 131},
            "C": {2: 12, 28: 142, 13: 67, 10: 52, 26: 132},
            "D": {2: 13, 28: 143, 13: 68, 10: 53, 26: 133},
            "E": {2: 14, 28: 144, 13: 69, 10: 54, 26: 134},
        }
    )
    y_expected = pd.Series({2: 0, 28: 1, 13: 3, 10: 0, 26: 2}, name="TARGET")
    return obj, X, y, X_expected, y_expected


@pytest.fixture
def data_ks():
    n_rows = 30
    n_cols = 5
    n_classes = 4
    n_samples = 5
    X = ks.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    np.random.seed(1)
    y = ks.Series(np.random.randint(0, n_classes, n_rows), name="TARGET")
    obj = UnsupervisedSampling(n_samples=n_samples)
    X_expected = pd.DataFrame(
        {
            "A": {0: 0, 7: 35, 8: 40, 18: 90, 21: 105, 20: 100},
            "B": {0: 1, 7: 36, 8: 41, 18: 91, 21: 106, 20: 101},
            "C": {0: 2, 7: 37, 8: 42, 18: 92, 21: 107, 20: 102},
            "D": {0: 3, 7: 38, 8: 43, 18: 93, 21: 108, 20: 103},
            "E": {0: 4, 7: 39, 8: 44, 18: 94, 21: 109, 20: 104},
        }
    )
    y_expected = pd.Series({0: 1, 7: 1, 8: 3, 18: 2, 21: 1, 20: 2}, name="TARGET")
    return obj, X, y, X_expected, y_expected


def test_pd(data):
    obj, X, y, X_expected, y_expected = data
    X_new, y_new = obj.transform(X, y)
    assert_frame_equal(X_new.sort_index(), X_expected.sort_index())
    assert_series_equal(y_new.sort_index(), y_expected.sort_index())


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, y, X_expected, y_expected = data_ks
    X_new, y_new = obj.transform(X, y)
    assert X_new.to_pandas().shape[0] + 1 == X_expected.shape[0]
    assert y_new.to_pandas().shape[0] == y_new.shape[0]
    assert X_new.to_pandas().shape[1] == X_expected.shape[1]


def test_init():
    with pytest.raises(TypeError):
        _ = UnsupervisedSampling(n_samples="a")


def test_no_sampling():
    X = pd.DataFrame({"A": [0, 1, 2]})
    y = pd.Series([0, 0, 1], name="TARGET")
    obj = UnsupervisedSampling(n_samples=5)
    assert_frame_equal(X, obj.transform(X, y)[0])
    assert_series_equal(y, obj.transform(X, y)[1])
