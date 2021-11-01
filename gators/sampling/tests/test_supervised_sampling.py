# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.sampling import SupervisedSampling

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    n_rows = 30
    n_cols = 5
    n_samples = 5
    np.random.seed(1)
    X = pd.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    y = pd.Series(
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
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
        ],
        name="TARGET",
    )
    obj = SupervisedSampling(n_samples=n_samples)
    X_expected = pd.DataFrame(
        {
            "A": {2: 10, 8: 40, 12: 60, 18: 90, 27: 135, 22: 110, 29: 145},
            "B": {2: 11, 8: 41, 12: 61, 18: 91, 27: 136, 22: 111, 29: 146},
            "C": {2: 12, 8: 42, 12: 62, 18: 92, 27: 137, 22: 112, 29: 147},
            "D": {2: 13, 8: 43, 12: 63, 18: 93, 27: 138, 22: 113, 29: 148},
            "E": {2: 14, 8: 44, 12: 64, 18: 94, 27: 139, 22: 114, 29: 149},
        }
    )
    y_expected = pd.Series(
        {2: 0, 8: 0, 12: 1, 18: 1, 27: 2, 22: 2, 29: 3}, name="TARGET"
    )
    return obj, X, y, X_expected, y_expected


@pytest.fixture
def data_ks():
    n_rows = 30
    n_cols = 5
    n_samples = 5
    X = ks.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    y = ks.Series(
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
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
        ],
        name="TARGET",
    )
    np.random.seed(1)
    obj = SupervisedSampling(n_samples=n_samples)
    X_expected = pd.DataFrame(
        {
            "A": {0: 0, 7: 35, 8: 40, 18: 90, 21: 105, 20: 100, 29: 145},
            "B": {0: 1, 7: 36, 8: 41, 18: 91, 21: 106, 20: 101, 29: 146},
            "C": {0: 2, 7: 37, 8: 42, 18: 92, 21: 107, 20: 102, 29: 147},
            "D": {0: 3, 7: 38, 8: 43, 18: 93, 21: 108, 20: 103, 29: 148},
            "E": {0: 4, 7: 39, 8: 44, 18: 94, 21: 109, 20: 104, 29: 149},
        }
    )
    y_expected = pd.Series(
        {0: 0, 7: 0, 8: 0, 18: 1, 21: 2, 20: 2, 29: 3}, name="TARGET"
    )
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
    assert_frame_equal(X_new.to_pandas().sort_index(), X_expected.sort_index())
    assert_series_equal(y_new.to_pandas().sort_index(), y_expected.sort_index())


def test_init():
    with pytest.raises(TypeError):
        _ = SupervisedSampling(n_samples="a")
