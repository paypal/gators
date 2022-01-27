import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_low_cardinality import DropLowCardinality


@pytest.fixture
def data():
    X = pd.DataFrame({"A": list("abc"), "B": list("abb"), "C": list("ccc")})
    X_expected = pd.DataFrame({"A": list("abc"), "B": list("abb")})
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_drop():
    X = pd.DataFrame({"A": list("abc"), "B": list("abb"), "C": list("ccd")})
    X_expected = X.copy()
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_object():
    X = pd.DataFrame(
        {"A": [0.0, 1.0, 2.0], "B": [3.0, 4.0, 5.0], "C": [6.0, 7.0, 8.0]}, dtype=float
    )
    X_expected = X.copy()
    obj = DropLowCardinality(min_categories=2).fit(X)
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


def test_no_drop_pd(data_no_drop):
    obj, X, X_expected = data_no_drop
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_drop_pd_np(data_no_drop):
    obj, X, X_expected = data_no_drop
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_object_pd(data_no_object):
    obj, X, X_expected = data_no_object
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_object_pd_np(data_no_object):
    obj, X, X_expected = data_no_object
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_drop_high_cardinality_init():
    with pytest.raises(TypeError):
        _ = DropLowCardinality(min_categories="q")


def test_no_obj():
    X = pd.DataFrame({"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8]})
    assert_frame_equal(X, DropLowCardinality(min_categories=2).fit_transform(X))
