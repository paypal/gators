# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import Extract

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["0", "1Q", "1QQ"]
    X["s"] = ["0", "W2", "W2W"]
    X["d"] = ["0", "Q", ""]
    obj = Extract(columns=list("asd"), i_min_vec=[0, 1, 2], i_max_vec=[1, 2, 3]).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__substring_0_to_1",
        "s__substring_1_to_2",
        "d__substring_2_to_3",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", "0", "MISSING", "MISSING"],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", "1", "2", "MISSING"],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", "1", "2", "MISSING"],
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
    obj = Extract(columns=list("asd"), i_min_vec=[0, 1, 2], i_max_vec=[1, 2, 3]).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__substring_0_to_1",
        "s__substring_1_to_2",
        "d__substring_2_to_3",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", "0", "MISSING", "MISSING"],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", "1", "2", "MISSING"],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", "1", "2", "MISSING"],
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


def test_drop_low_cardinality_init(data):
    with pytest.raises(TypeError):
        _ = Extract(columns="x", i_min_vec=[0, 1], i_max_vec=[1, 2])
    with pytest.raises(TypeError):
        _ = Extract(columns=["a", "s"], i_min_vec=0, i_max_vec=[1, 2])
    with pytest.raises(TypeError):
        _ = Extract(columns=["a", "s"], i_min_vec=[0, 1], i_max_vec=0)
    with pytest.raises(ValueError):
        _ = Extract(columns=["a", "s"], i_min_vec=[0], i_max_vec=[1, 2])
    with pytest.raises(ValueError):
        _ = Extract(columns=["a", "s"], i_min_vec=[0, 1], i_max_vec=[1])
