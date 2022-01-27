# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.model_building.train_test_split import TrainTestSplit


@pytest.fixture()
def data_ordered_ks():
    X = ks.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name="TARGET")
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="ordered")
    X_train_expected = pd.DataFrame(
        {
            "A": [0, 5, 10, 15],
            "B": [1, 6, 11, 16],
            "C": [2, 7, 12, 17],
            "D": [3, 8, 13, 18],
            "E": [4, 9, 14, 19],
        }
    )
    X_test_expected = pd.DataFrame(
        {
            "A": [20, 25, 30, 35],
            "B": [21, 26, 31, 36],
            "C": [22, 27, 32, 37],
            "D": [23, 28, 33, 38],
            "E": [24, 29, 34, 39],
        },
        index=[4, 5, 6, 7],
    )
    y_train_expected = pd.Series([0, 1, 2, 0], name="TARGET")
    y_test_expected = pd.Series([1, 2, 0, 1], name="TARGET", index=[4, 5, 6, 7])
    return (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    )


@pytest.fixture()
def data_random_ks():
    X = ks.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE"))
    y_name = "TARGET"
    y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name=y_name)
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="random", random_state=0)
    return obj, X, y


@pytest.fixture()
def data_stratified_ks():
    X = ks.DataFrame(np.arange(40).reshape(10, 4), columns=list("ABCD"))
    y = ks.Series([0, 1, 2, 0, 1, 2, 0, 0, 0, 0], name="TARGET")
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="stratified", random_state=0)
    return obj, X, y


@pytest.mark.koalas
def test_ordered_ks(data_ordered_ks):
    (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    ) = data_ordered_ks
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert_frame_equal(X_train.to_pandas(), X_train_expected)
    assert_frame_equal(X_test.to_pandas(), X_test_expected)
    assert_series_equal(y_train.to_pandas(), y_train_expected)
    assert_series_equal(y_test.to_pandas(), y_test_expected)


@pytest.mark.koalas
def test_random_ks(data_random_ks):
    obj, X, y = data_random_ks
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert X_train.shape == (4, 5)
    assert X_test.shape == (4, 5)
    assert y_train.shape == (4,)
    assert y_test.shape == (4,)
    assert len(set(X_train.index.tolist() + X_test.index.tolist())) == 8


@pytest.mark.koalas
def test_stratified_ks(data_stratified_ks):
    obj, X, y = data_stratified_ks
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert len(set(X_train.index.tolist() + X_test.index.tolist())) == 10
