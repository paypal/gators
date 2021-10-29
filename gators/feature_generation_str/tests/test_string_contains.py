# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import StringContains

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1Q", "1QQ"]
    X["s"] = ["0", "W2", "W2W"]
    X["d"] = ["0", "Q", ""]

    obj = StringContains(columns=list("asd"), contains_vec=["1", "2", "0"]).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__contains_1",
        "s__contains_2",
        "d__contains_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", 1.0, 1.0, 0.0],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1Q", "1QQ"]
    X["s"] = ["0", "W2", "W2W"]
    X["d"] = ["0", "Q", ""]

    obj = StringContains(columns=list("asd"), contains_vec=["1", "2", "0"]).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__contains_1",
        "s__contains_2",
        "d__contains_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", 1.0, 1.0, 0.0],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


@pytest.fixture
def data_with_names():
    X = pd.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1Q", "1QQ"]
    X["s"] = ["0", "W2", "W2W"]
    X["d"] = ["0", "Q", ""]

    obj = StringContains(
        columns=list("asd"),
        contains_vec=["1", "2", "0"],
        column_names=["a_with_1", "s_with_2", "d_with_0"],
    ).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a_with_1",
        "s_with_2",
        "d_with_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", 1.0, 1.0, 0.0],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


@pytest.fixture
def data_with_names_ks():
    X = ks.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1Q", "1QQ"]
    X["s"] = ["0", "W2", "W2W"]
    X["d"] = ["0", "Q", ""]

    obj = StringContains(
        columns=list("asd"),
        contains_vec=["1", "2", "0"],
        column_names=["a_with_1", "s_with_2", "d_with_0"],
    ).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a_with_1",
        "s_with_2",
        "d_with_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", 1.0, 1.0, 0.0],
        ],
        columns=columns_expected,
    )
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


def test_names_pd(data_with_names):
    obj, X, X_expected = data_with_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks(data_with_names_ks):
    obj, X, X_expected = data_with_names_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_names_pd_np(data_with_names):
    obj, X, X_expected = data_with_names
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_names_ks_np(data_with_names_ks):
    obj, X, X_expected = data_with_names_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = StringContains(
            columns="x", contains_vec=["z", "x"], column_names=["aa", "ss"]
        )
    with pytest.raises(TypeError):
        _ = StringContains(
            columns=["a", "s"], contains_vec="x", column_names=["aa", "ss"]
        )
    with pytest.raises(TypeError):
        _ = StringContains(
            columns=["a", "s"], contains_vec=["z", "x"], column_names="x"
        )
    with pytest.raises(ValueError):
        _ = StringContains(
            columns=["a", "s"], contains_vec=["z"], column_names=["aa", "ss"]
        )
    with pytest.raises(ValueError):
        _ = StringContains(
            columns=["a", "s"], contains_vec=["z", "x"], column_names=["aa"]
        )
