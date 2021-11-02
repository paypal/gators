# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.elementary_arithmethics import ElementaryArithmetics


@pytest.fixture
def data_add():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "A-2xB": [-2.0, -5.0, -8.0],
            "A-2xC": [-4.0, -7.0, -10.0],
        }
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"), columns_b=list("BC"), coef=-2.0, operator="+"
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_object_add():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
                "D": ["a", "b", "c"],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "D": ["a", "b", "c"],
            "A-2xB": [-2.0, -5.0, -8.0],
            "A-2xC": [-4.0, -7.0, -10.0],
        }
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"),
        columns_b=list("BC"),
        coef=-2.0,
        operator="+",
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_name_add():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "Aplus2B": [2.0, 11.0, 20.0],
            "Aplus2C": [4.0, 13.0, 22.0],
        }
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"),
        columns_b=list("BC"),
        coef=2.0,
        operator="+",
        column_names=["Aplus2B", "Aplus2C"],
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_mult():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "A*B": [0.0, 12.0, 42.0],
            "A*C": [0.0, 15.0, 48.0],
        }
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"), columns_b=list("BC"), operator="*"
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_div():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 3.0, 6.0],
                "B": [1.0, 4.0, 7.0],
                "C": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "A/B": [0.0, 0.75, 0.85714286],
            "A/C": [0.0, 0.59999988, 0.7499999],
        }
    )
    obj = ElementaryArithmetics(
        columns_a=list("AA"), columns_b=list("BC"), operator="/"
    ).fit(X)
    return obj, X, X_expected


def test_add_dd(data_add):
    obj, X, X_expected = data_add
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_add_dd_np(data_add):
    obj, X, X_expected = data_add
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_object_add_dd(data_object_add):
    obj, X, X_expected = data_object_add
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_object_add_dd_np(data_object_add):
    obj, X, X_expected = data_object_add
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_mult_dd(data_mult):
    obj, X, X_expected = data_mult
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_mult_dd_np(data_mult):
    obj, X, X_expected = data_mult
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_div_dd(data_div):
    obj, X, X_expected = data_div
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_div_dd_np(data_div):
    obj, X, X_expected = data_div
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_name_add_dd(data_name_add):
    obj, X, X_expected = data_name_add
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_name_add_dd_np(data_name_add):
    obj, X, X_expected = data_name_add
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
