# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.is_equal import IsEqual


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_obj():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["a", "b", "c"],
                "B": ["a", "f", "e"],
                "C": ["a", "p", "d"],
                "D": [1.0, 2.0, 3.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["a", "f", "e"],
            "C": ["a", "p", "d"],
            "D": [1.0, 2.0, 3.0],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names():
    X = dd.from_pandas(
        pd.DataFrame(
            {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A==B": [1.0, 0.0, 0.0],
            "A==C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(
        columns_a=list("AA"), columns_b=list("BC"), column_names=["A==B", "A==C"]
    ).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)


def test_obj(data_obj):
    obj, X, X_expected = data_obj
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_obj_np(data_obj):
    obj, X, X_expected = data_obj
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_dd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_names_dd_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)
