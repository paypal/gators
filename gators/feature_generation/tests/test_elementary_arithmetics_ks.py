# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.elementary_arithmethics import ElementaryArithmetics


@pytest.fixture
def data_add_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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
def data_object_add_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "D": ["a", "b", "c"],
        }
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
def data_name_add_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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
def data_mult_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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
def data_div_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
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


@pytest.mark.koalas
def test_add_ks(data_add_ks):
    obj, X, X_expected = data_add_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_add_ks_np(data_add_ks):
    obj, X, X_expected = data_add_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_object_add_ks_ks(data_object_add_ks):
    obj, X, X_expected = data_object_add_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_object_add_ks_np_ks(data_object_add_ks):
    obj, X, X_expected = data_object_add_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_mult_ks(data_mult_ks):
    obj, X, X_expected = data_mult_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_mult_ks_np(data_mult_ks):
    obj, X, X_expected = data_mult_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_div_ks(data_div_ks):
    obj, X, X_expected = data_div_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_div_ks_np(data_div_ks):
    obj, X, X_expected = data_div_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_name_add_ks_ks(data_name_add_ks):
    obj, X, X_expected = data_name_add_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_name_add_ks_np_ks(data_name_add_ks):
    obj, X, X_expected = data_name_add_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
