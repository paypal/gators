import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_low_cardinality import DropLowCardinality


@pytest.fixture
def data_ks():
    X = ks.DataFrame({"A": list("abc"), "B": list("abb"), "C": list("ccc")})
    X_expected = pd.DataFrame({"A": list("abc"), "B": list("abb")})
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_drop_ks():
    X = ks.DataFrame({"A": list("abc"), "B": list("abb"), "C": list("ccd")})
    X_expected = X.to_pandas().copy()
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_object_ks():
    X = ks.DataFrame(
        {"A": [0.0, 1.0, 2.0], "B": [3.0, 4.0, 5.0], "C": [6.0, 7.0, 8.0]}, dtype=float
    )
    X_expected = X.to_pandas().copy()
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_drop_ks(data_no_drop_ks):
    obj, X, X_expected = data_no_drop_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_no_drop_ks_np(data_no_drop_ks):
    obj, X, X_expected = data_no_drop_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_object_ks(data_no_object_ks):
    obj, X, X_expected = data_no_object_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_no_object_ks_np(data_no_object_ks):
    obj, X, X_expected = data_no_object_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_drop_high_cardinality_init():
    with pytest.raises(TypeError):
        _ = DropLowCardinality(min_categories="q")
