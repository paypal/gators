import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_high_cardinality import DropHighCardinality


@pytest.fixture
def data_pd():
    X = pd.DataFrame(
        {"A": list("qwe"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = pd.DataFrame({"B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]})
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_drop_pd():
    X = pd.DataFrame(
        {"A": list("qww"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = X.copy()
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_object_pd():
    X = pd.DataFrame({"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8]})
    X_expected = X.copy()
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


def test_pd(data_pd):
    obj, X, X_expected = data_pd
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data_pd):
    obj, X, X_expected = data_pd
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_drop_pd(data_no_drop_pd):
    obj, X, X_expected = data_no_drop_pd
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_drop_pd_np(data_no_drop_pd):
    obj, X, X_expected = data_no_drop_pd
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_object_pd(data_no_object_pd):
    obj, X, X_expected = data_no_object_pd
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_object_pd_np(data_no_object_pd):
    obj, X, X_expected = data_no_object_pd
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = DropHighCardinality(max_categories="q")
