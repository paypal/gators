# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import LowerCase


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": ["q", "qq", "QQq"],
            "E": ["w", "WW", "WWw"],
            "F": ["nan", None, ""],
        }
    )

    obj = LowerCase(columns=list("DEF"), inplace=False).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": ["q", "qq", "QQq"],
            "E": ["w", "WW", "WWw"],
            "F": ["nan", None, ""],
            "D__lower": ["q", "qq", "qqq"],
            "E__lower": ["w", "ww", "www"],
            "F__lower": [None, None, ""],
        }
    )
    return obj, X, X_expected


@pytest.fixture
def data_inplace():
    X = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": ["q", "qq", "QQq"],
            "E": ["w", "WW", "WWw"],
            "F": ["nan", None, ""],
        }
    )

    obj = LowerCase(columns=list("DEF"), inplace=True).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": ["q", "qq", "qqq"],
            "E": ["w", "ww", "www"],
            "F": [None, None, ""],
        }
    )
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_inplace_pd(data_inplace):
    obj, X, X_expected = data_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_inplace_pd_np(data_inplace):
    obj, X, X_expected = data_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = LowerCase(columns="x")
    with pytest.raises(ValueError):
        _ = LowerCase(columns=[])
    with pytest.raises(TypeError):
        _ = LowerCase(columns=["x"], inplace="x")
