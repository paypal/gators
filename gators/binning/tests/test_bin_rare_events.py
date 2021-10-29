# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.bin_rare_events import BinRareEvents

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["w", "z", "q", "q", "q", "z"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "e", "d", "d", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["OTHERS", "OTHERS", "q", "q", "q", "OTHERS"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "OTHERS", "OTHERS", "OTHERS", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareEvents(min_ratio=0.5).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_1object():
    X = pd.DataFrame({"A": ["w", "z", "q", "q", "q", "z"], "D": [1, 2, 3, 4, 5, 6]})
    X_expected = pd.DataFrame(
        {"A": ["OTHERS", "OTHERS", "q", "q", "q", "OTHERS"], "D": [1, 2, 3, 4, 5, 6]}
    )
    obj = BinRareEvents(min_ratio=0.5).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_all_others():
    X = pd.DataFrame(
        {
            "A": ["w", "z", "q", "q", "q", "z"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "e", "d", "d", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "B": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "C": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareEvents(min_ratio=1.0).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_other():
    X = pd.DataFrame(
        {
            "A": ["w", "z", "q", "q", "q", "z"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "e", "d", "d", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareEvents(min_ratio=0.0).fit(X)
    obj = BinRareEvents(min_ratio=0.0).fit(X)
    return obj, X, X.copy()


@pytest.fixture
def data_num():
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [1, 2, 3, 4, 5, 6]})
    obj = BinRareEvents(min_ratio=1.0).fit(X)
    return obj, X, X.copy()


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["w", "z", "q", "q", "q", "z"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "e", "d", "d", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["OTHERS", "OTHERS", "q", "q", "q", "OTHERS"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "OTHERS", "OTHERS", "OTHERS", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareEvents(min_ratio=0.5).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_1object_ks():
    X = ks.DataFrame({"A": ["w", "z", "q", "q", "q", "z"], "D": [1, 2, 3, 4, 5, 6]})
    X_expected = pd.DataFrame(
        {"A": ["OTHERS", "OTHERS", "q", "q", "q", "OTHERS"], "D": [1, 2, 3, 4, 5, 6]}
    )
    obj = BinRareEvents(min_ratio=0.5).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_all_others_ks():
    X = ks.DataFrame(
        {
            "A": ["w", "z", "q", "q", "q", "z"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "e", "d", "d", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "B": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "C": ["OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS", "OTHERS"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareEvents(min_ratio=1.0).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_other_ks():
    X = ks.DataFrame(
        {
            "A": ["w", "z", "q", "q", "q", "z"],
            "B": ["x", "x", "w", "w", "w", "x"],
            "C": ["c", "c", "e", "d", "d", "c"],
            "D": [1, 2, 3, 4, 5, 6],
        }
    )
    obj = BinRareEvents(min_ratio=0.0).fit(X)
    obj = BinRareEvents(min_ratio=0.0).fit(X)
    return obj, X, X.to_pandas().copy()


@pytest.fixture
def data_num_ks():
    X = ks.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [1, 2, 3, 4, 5, 6]})
    obj = BinRareEvents(min_ratio=1.0).fit(X)
    return obj, X, X.to_pandas().copy()


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
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected.astype(object))


def test_num_pd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_num_ks(data_num_ks):
    obj, X, X_expected = data_num_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_num_pd_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_num_ks_np(data_num_ks):
    obj, X, X_expected = data_num_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_other_pd(data_no_other):
    obj, X, X_expected = data_no_other
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_other_ks(data_no_other_ks):
    obj, X, X_expected = data_no_other_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_no_other_pd_np(data_no_other):
    obj, X, X_expected = data_no_other
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_other_ks_np(data_no_other_ks):
    obj, X, X_expected = data_no_other_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_1object_pd(data_1object):
    obj, X, X_expected = data_1object
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_1object_ks(data_1object_ks):
    obj, X, X_expected = data_1object_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_1object_pd_np(data_1object):
    obj, X, X_expected = data_1object
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_1object_ks_np(data_1object_ks):
    obj, X, X_expected = data_1object_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_all_others_pd(data_all_others):
    obj, X, X_expected = data_all_others
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_all_others_ks(data_all_others_ks):
    obj, X, X_expected = data_all_others_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_all_others_pd_np(data_all_others):
    obj, X, X_expected = data_all_others
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_all_others_ks_np(data_all_others_ks):
    obj, X, X_expected = data_all_others_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = BinRareEvents(min_ratio=1)
