# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import SplitExtract

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1*Q", "1Q*QQ"]
    X["s"] = ["0", "W*2", "W2*WW"]
    X["d"] = ["0", "Q*", "qwert*"]

    obj = SplitExtract(list("asd"), list("***"), [1, 1, 0]).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__split_by_*_idx_1",
        "s__split_by_*_idx_1",
        "d__split_by_*_idx_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", "MISSING", "MISSING", "0"],
            [0.0, 0.0, 0.0, "1*Q", "W*2", "Q*", "Q", "2", "Q"],
            [0.0, 0.0, 0.0, "1Q*QQ", "W2*WW", "qwert*", "QQ", "WW", "qwert"],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1*Q", "1Q*QQ"]
    X["s"] = ["0", "W*2", "W2*WW"]
    X["d"] = ["0", "Q*", "qwert*"]

    obj = SplitExtract(list("asd"), list("***"), [1, 1, 0]).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__split_by_*_idx_1",
        "s__split_by_*_idx_1",
        "d__split_by_*_idx_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", "MISSING", "MISSING", "0"],
            [0.0, 0.0, 0.0, "1*Q", "W*2", "Q*", "Q", "2", "Q"],
            [0.0, 0.0, 0.0, "1Q*QQ", "W2*WW", "qwert*", "QQ", "WW", "qwert"],
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


def test_init():
    with pytest.raises(TypeError):
        _ = SplitExtract(columns="x", str_split_vec=["+", "-"], idx_split_vec=[1, 2])
    with pytest.raises(TypeError):
        _ = SplitExtract(columns=["a", "s"], str_split_vec="+", idx_split_vec=[1, 2])
    with pytest.raises(TypeError):
        _ = SplitExtract(columns=["a", "s"], str_split_vec=["+", "-"], idx_split_vec=0)
    with pytest.raises(ValueError):
        _ = SplitExtract(columns=["a", "s"], str_split_vec=["+"], idx_split_vec=[1, 2])
    with pytest.raises(ValueError):
        _ = SplitExtract(
            columns=["a", "s"], str_split_vec=["+", "-"], idx_split_vec=[1]
        )
