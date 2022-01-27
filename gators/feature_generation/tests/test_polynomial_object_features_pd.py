# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.polynomial_object_features import (
    PolynomialObjectFeatures,
)


@pytest.fixture
def data():
    X = pd.DataFrame(
        {"A": [None, "b", "c"], "B": ["z", "a", "a"], "C": ["c", "d", "d"]}
    )
    X_expected = pd.DataFrame(
        {
            "A": [None, "b", "c"],
            "B": ["z", "a", "a"],
            "C": ["c", "d", "d"],
            "A__B": ["z", "ba", "ca"],
            "A__C": ["c", "bd", "cd"],
            "B__C": ["zc", "ad", "ad"],
            "A__B__C": ["zc", "bad", "cad"],
        }
    )
    obj = PolynomialObjectFeatures(columns=["A", "B", "C"], degree=3).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_degree1():
    X = pd.DataFrame(
        {"A": [None, "b", "c"], "B": ["z", "a", "a"], "C": ["c", "d", "d"]}
    )
    X_expected = pd.DataFrame(
        {
            "A": [None, "b", "c"],
            "B": ["z", "a", "a"],
            "C": ["c", "d", "d"],
        }
    )
    obj = PolynomialObjectFeatures(columns=["A", "B", "C"], degree=1).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_pd(data_degree1):
    obj, X, X_expected = data_degree1
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data_degree1):
    obj, X, X_expected = data_degree1
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = PolynomialObjectFeatures(columns="A", degree=2)
    with pytest.raises(ValueError):
        _ = PolynomialObjectFeatures(columns=["A"], degree=-1)
    with pytest.raises(TypeError):
        _ = PolynomialObjectFeatures(columns=["A", "B"], degree=1.1)
    with pytest.raises(ValueError):
        _ = PolynomialObjectFeatures(columns=[], degree=2)
