# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import StringLength

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["Q", "QQ", "QQQ"]
    X["s"] = ["W", "WW", "WWW"]
    X["d"] = ["nan", None, ""]

    obj = StringLength(columns=list("asd")).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__length",
        "s__length",
        "d__length",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "Q", "W", "nan", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "QQ", "WW", None, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, "QQQ", "WWW", "", 3.0, 3.0, 0.0],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.zeros((3, 3)), columns=list("qwe"))
    X["a"] = ["Q", "QQ", "QQQ"]
    X["s"] = ["W", "WW", "WWW"]
    X["d"] = ["nan", None, ""]

    obj = StringLength(columns=list("asd")).fit(X)
    columns_expected = [
        "q",
        "w",
        "e",
        "a",
        "s",
        "d",
        "a__length",
        "s__length",
        "d__length",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "Q", "W", "nan", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "QQ", "WW", None, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, "QQQ", "WWW", "", 3.0, 3.0, 0.0],
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
        _ = StringLength(columns="x")
    with pytest.raises(ValueError):
        _ = StringLength(columns=[])
