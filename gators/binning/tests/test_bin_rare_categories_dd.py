# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.bin_rare_categories import BinRareCategories


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["w", "z", "q", "q", "q", "z"],
                "B": ["x", "x", "w", "w", "w", "x"],
                "C": ["c", "c", "e", "d", "d", "c"],
                "D": [1, 2, 3, 4, 5, 6],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["OTHERS", "OTHERS", "q", "q", "q", "OTHERS"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "OTHERS", "OTHERS", "OTHERS", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareCategories(min_ratio=0.5).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_all_others():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["w", "z", "q", "q", "q", "z"],
                "B": ["x", "x", "w", "w", "w", "x"],
                "C": ["c", "c", "e", "d", "d", "c"],
                "D": [1, 2, 3, 4, 5, 6],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "A": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "B": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "C": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareCategories(min_ratio=1.0).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_other():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["w", "z", "q", "q", "q", "z"],
                "B": ["x", "x", "w", "w", "w", "x"],
                "C": ["c", "c", "e", "d", "d", "c"],
                "D": [1, 2, 3, 4, 5, 6],
            }
        ),
        npartitions=1,
    )
    obj = BinRareCategories(min_ratio=0.0).fit(X)
    obj = BinRareCategories(min_ratio=0.0).fit(X)
    return obj, X, X.compute().copy()


@pytest.fixture
def data_num():
    X = dd.from_pandas(
        pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [1, 2, 3, 4, 5, 6]}), npartitions=1
    )
    obj = BinRareCategories(min_ratio=1.0).fit(X)
    return obj, X, X.compute().copy()


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected.astype(object))


def test_num_dd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_num_dd_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_other_dd(data_no_other):
    obj, X, X_expected = data_no_other
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_no_other_dd_np(data_no_other):
    obj, X, X_expected = data_no_other
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_all_others_dd(data_all_others):
    obj, X, X_expected = data_all_others
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_all_others_dd_np(data_all_others):
    obj, X, X_expected = data_all_others
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
