# License: Apache-2.0
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from gators.feature_generation.elementary_arithmethics import ElementaryArithmetics


@pytest.fixture
def data_add():
    X = pd.DataFrame(
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
def data_object_add():
    X = pd.DataFrame(
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
def data_name_add():
    X = pd.DataFrame(
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
def data_mult():
    X = pd.DataFrame(
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
def data_div():
    X = pd.DataFrame(
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


def test_add_pd(data_add):
    obj, X, X_expected = data_add
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_add_pd_np(data_add):
    obj, X, X_expected = data_add
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_object_add_pd(data_object_add):
    obj, X, X_expected = data_object_add
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_object_add_pd_np(data_object_add):
    obj, X, X_expected = data_object_add
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_mult_pd(data_mult):
    obj, X, X_expected = data_mult
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_mult_pd_np(data_mult):
    obj, X, X_expected = data_mult
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_div_pd(data_div):
    obj, X, X_expected = data_div
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_div_pd_np(data_div):
    obj, X, X_expected = data_div
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_name_add_pd(data_name_add):
    obj, X, X_expected = data_name_add
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_name_add_pd_np(data_name_add):
    obj, X, X_expected = data_name_add
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_operator_str():
    assert ElementaryArithmetics.get_str_operator("+", coef=1) == "+"
    assert ElementaryArithmetics.get_str_operator("*", coef=2) == "*"
    assert ElementaryArithmetics.get_str_operator("+", coef=2.0) == "+2x"


def test_init():
    with pytest.raises(TypeError):
        _ = ElementaryArithmetics(
            columns_a="A", columns_b=["A"], operator="+", column_names=["2A"]
        )
    with pytest.raises(TypeError):
        _ = ElementaryArithmetics(
            columns_a=["A"], columns_b="A", operator="+", column_names=["2A"]
        )
    with pytest.raises(TypeError):
        _ = ElementaryArithmetics(
            columns_a=["A"], columns_b=["A"], operator=0, column_names=["2A"]
        )
    with pytest.raises(ValueError):
        _ = ElementaryArithmetics(
            columns_a=["A"], columns_b=["A"], operator="z", column_names=["2A"]
        )
    with pytest.raises(ValueError):
        _ = ElementaryArithmetics(
            columns_a=["A", "B"], columns_b=["A"], operator="+", column_names=["2A"]
        )
    with pytest.raises(ValueError):
        _ = ElementaryArithmetics(
            columns_a=[], columns_b=["A"], operator="+", column_names=["2A"]
        )
    with pytest.raises(ValueError):
        _ = ElementaryArithmetics(
            columns_a=["A"], columns_b=["A"], operator="+", column_names=["2A", "2A"]
        )
    with pytest.raises(TypeError):
        _ = ElementaryArithmetics(
            columns_a=["A"], columns_b=["A"], operator="+", coef="x"
        )
    with pytest.raises(TypeError):
        _ = ElementaryArithmetics(
            columns_a=["A"], columns_b=["A"], operator="+", column_names="x"
        )
