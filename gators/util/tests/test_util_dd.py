# License: Apache-2.0
import numpy as np
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.util import util


@pytest.fixture
def data():
    return dd.from_pandas(
        pd.DataFrame(
            {"A": 1.1, "B": 2.1, "C": 3.1, "D": 4, "E": 5, "F": "a"}, index=[0]
        ),
        npartitions=1,
    )


def test_get_bounds():
    assert np.allclose(util.get_bounds(np.float32), [-3.4028235e38, 3.4028235e38])
    assert np.allclose(util.get_bounds(np.int32), [-2147483648, 2147483647])
    assert np.allclose(util.get_bounds(np.int16), [-32768, 32767])


def test_get_datatype_columns_float_pd(data):
    X = data
    float_columns = util.get_datatype_columns(X, np.float64)
    assert float_columns == ["A", "B", "C"]


def test_get_datatype_columns_int_pd(data):
    X = data
    int_columns = util.get_datatype_columns(X, np.int64)
    assert int_columns == ["D", "E"]


def test_get_datatype_columns_object_pd(data):
    X = data
    object_columns = util.get_datatype_columns(X, object)
    assert object_columns == ["F"]


def test_concat_pd(data):
    X = data
    assert_frame_equal(
        util.get_function(X)
        .concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1)
        .compute(),
        pd.concat(
            [X.compute(), X[["A"]].compute().rename(columns={"A": "AA"})], axis=1
        ),
    )


def test_unique(data):
    X = data
    _ = util.get_function(X).nunique(X)
    _ = util.get_function(X).nunique(X["A"])
    assert True
