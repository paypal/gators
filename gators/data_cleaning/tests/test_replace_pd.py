import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.replace import Replace


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": list("abcd"),
            "B": list("abcd"),
            "C": [0, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["W", "X", "c", "d"],
            "B": ["a", "b", "Y", "Z"],
            "C": [0, 1, 2, 3],
        }
    )
    to_replace_dict = {"A": {"a": "W", "b": "X"}, "B": {"c": "Y", "d": "Z"}}
    obj = Replace(to_replace_dict=to_replace_dict).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_not_inplace():
    X = pd.DataFrame(
        {
            "A": list("abcd"),
            "B": list("abcd"),
            "C": [0, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A__replace": ["W", "X", "c", "d"],
            "B__replace": ["a", "b", "Y", "Z"],
        }
    )
    to_replace_dict = {"A": {"a": "W", "b": "X"}, "B": {"c": "Y", "d": "Z"}}
    obj = Replace(to_replace_dict=to_replace_dict, inplace=False).fit(X)
    return obj, X, pd.concat([X, X_expected], axis=1)


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_not_inplace_pd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_not_inplace_pd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_drop_columns_init(data):
    with pytest.raises(TypeError):
        _ = Replace(to_replace_dict="q")
    with pytest.raises(ValueError):
        _ = Replace(to_replace_dict={})
