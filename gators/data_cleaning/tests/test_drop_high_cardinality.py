import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_high_cardinality import DropHighCardinality


@pytest.fixture
def data():
    X = pd.DataFrame(
        {"A": list("qwe"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = pd.DataFrame({"B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]})
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {"A": list("qwe"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = pd.DataFrame({"B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]})
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_drop():
    X = pd.DataFrame(
        {"A": list("qww"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = X.copy()
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_drop_ks():
    X = ks.DataFrame(
        {"A": list("qww"), "B": list("ass"), "C": list("zxx"), "D": [0, 1, 2]}
    )
    X_expected = X.to_pandas().copy()
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_object():
    X = pd.DataFrame({"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8]})
    X_expected = X.copy()
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_object_ks():
    X = ks.DataFrame({"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8]})
    X_expected = X.to_pandas().copy()
    obj = DropHighCardinality(max_categories=2).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_drop_pd(data_no_drop):
    obj, X, X_expected = data_no_drop
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_drop_ks(data_no_drop_ks):
    obj, X, X_expected = data_no_drop_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_no_drop_pd_np(data_no_drop):
    obj, X, X_expected = data_no_drop
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_drop_ks_np(data_no_drop_ks):
    obj, X, X_expected = data_no_drop_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_object_pd(data_no_object):
    obj, X, X_expected = data_no_object
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_object_ks(data_no_object_ks):
    obj, X, X_expected = data_no_object_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_no_object_pd_np(data_no_object):
    obj, X, X_expected = data_no_object
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_object_ks_np(data_no_object_ks):
    obj, X, X_expected = data_no_object_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_object_columns_empty(data_no_drop):
    X = pd.DataFrame({"A": [1, 2, 3]})
    X_new = DropHighCardinality(max_categories=2).fit_transform(X.copy())
    assert_frame_equal(X_new, X)


def test_init():
    with pytest.raises(TypeError):
        _ = DropHighCardinality(max_categories="q")
