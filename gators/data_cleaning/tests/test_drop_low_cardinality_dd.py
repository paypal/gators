import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_low_cardinality import DropLowCardinality


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame({"A": list("abc"), "B": list("abb"), "C": list("ccc")}),
        npartitions=1,
    )
    X_expected = pd.DataFrame({"A": list("abc"), "B": list("abb")})
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_drop():
    X = dd.from_pandas(
        pd.DataFrame({"A": list("abc"), "B": list("abb"), "C": list("ccd")}),
        npartitions=1,
    )
    X_expected = X.compute().copy()
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_object():
    X = dd.from_pandas(
        pd.DataFrame(
            {"A": [0.0, 1.0, 2.0], "B": [3.0, 4.0, 5.0], "C": [6.0, 7.0, 8.0]},
            dtype=float,
        ),
        npartitions=1,
    )
    X_expected = X.compute().copy()
    obj = DropLowCardinality(min_categories=2).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute()
    X_new[["A", "B"]] = X_new[["A", "B"]].astype(object)
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_drop_dd(data_no_drop):
    obj, X, X_expected = data_no_drop
    X_new = obj.transform(X).compute()
    assert_frame_equal(X_new, X_expected)


def test_no_drop_dd_np(data_no_drop):
    obj, X, X_expected = data_no_drop
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_object_dd(data_no_object):
    obj, X, X_expected = data_no_object
    X_new = obj.transform(X).compute()
    assert_frame_equal(X_new, X_expected)


def test_no_object_dd_np(data_no_object):
    obj, X, X_expected = data_no_object
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_drop_high_cardinality_init():
    with pytest.raises(TypeError):
        _ = DropLowCardinality(min_categories="q")


def test_no_obj():
    X = pd.DataFrame({"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8]})
    assert_frame_equal(X, DropLowCardinality(min_categories=2).fit_transform(X))
