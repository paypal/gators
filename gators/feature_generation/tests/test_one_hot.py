# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.one_hot import OneHot

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
    X_expected = pd.DataFrame(
        {
            "A": {0: "q", 1: "q", 2: "a"},
            "B": {0: "w", 1: "w", 2: "s"},
            "C": {0: "e", 1: "r", 2: "d"},
            "A__onehot__q": {0: True, 1: True, 2: False},
            "A__onehot__a": {0: False, 1: False, 2: True},
            "B__onehot__w": {0: True, 1: True, 2: False},
            "B__onehot__s": {0: False, 1: False, 2: True},
            "C__onehot__e": {0: True, 1: False, 2: False},
            "C__onehot__d": {0: False, 1: False, 2: True},
        }
    )
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names():
    X = pd.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
    X_expected = pd.DataFrame(
        {
            "A": {0: "q", 1: "q", 2: "a"},
            "B": {0: "w", 1: "w", 2: "s"},
            "C": {0: "e", 1: "r", 2: "d"},
            "Aq": {0: True, 1: True, 2: False},
            "Aa": {0: False, 1: False, 2: True},
            "Bw": {0: True, 1: True, 2: False},
            "Bs": {0: False, 1: False, 2: True},
            "Ce": {0: True, 1: False, 2: False},
            "Cd": {0: False, 1: False, 2: True},
        }
    )
    column_names = ["Aq", "Aa", "Bw", "Bs", "Ce", "Cd"]
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict, column_names=column_names).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
    X_expected = pd.DataFrame(
        {
            "A": {0: "q", 1: "q", 2: "a"},
            "B": {0: "w", 1: "w", 2: "s"},
            "C": {0: "e", 1: "r", 2: "d"},
            "A__onehot__q": {0: True, 1: True, 2: False},
            "A__onehot__a": {0: False, 1: False, 2: True},
            "B__onehot__w": {0: True, 1: True, 2: False},
            "B__onehot__s": {0: False, 1: False, 2: True},
            "C__onehot__e": {0: True, 1: False, 2: False},
            "C__onehot__d": {0: False, 1: False, 2: True},
        }
    )
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names_ks():
    X = ks.DataFrame(np.array(list("qweqwrasd")).reshape(3, 3), columns=list("ABC"))
    X_expected = pd.DataFrame(
        {
            "A": {0: "q", 1: "q", 2: "a"},
            "B": {0: "w", 1: "w", 2: "s"},
            "C": {0: "e", 1: "r", 2: "d"},
            "Aq": {0: True, 1: True, 2: False},
            "Aa": {0: False, 1: False, 2: True},
            "Bw": {0: True, 1: True, 2: False},
            "Bs": {0: False, 1: False, 2: True},
            "Ce": {0: True, 1: False, 2: False},
            "Cd": {0: False, 1: False, 2: True},
        }
    )
    column_names = ["Aq", "Aa", "Bw", "Bs", "Ce", "Cd"]
    categories_dict = {"A": ["q", "a"], "B": ["w", "s"], "C": ["e", "d"]}
    obj = OneHot(categories_dict=categories_dict, column_names=column_names).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.iloc[:, -3:], X_expected.iloc[:, -3:])


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas().iloc[:, -3:], X_expected.iloc[:, -3:])


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_pd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_pd_names_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks_names_np(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = OneHot(categories_dict=0)
    with pytest.raises(TypeError):
        _ = OneHot(categories_dict={"a": ["x"]}, column_names=0)
    with pytest.raises(ValueError):
        _ = OneHot(categories_dict={"a": ["x"]}, column_names=["aa", "bb"])
