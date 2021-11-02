# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.model_building.train_test_split import TrainTestSplit


@pytest.fixture()
def data_ordered():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE")), npartitions=1
    )
    y = dd.from_pandas(
        pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name="TARGET"), npartitions=1
    )
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
def data_random():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(40).reshape(8, 5), columns=list("ABCDE")), npartitions=1
    )
    y = dd.from_pandas(
        pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name="TARGET"), npartitions=1
    )
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="random", random_state=0)
    return obj, X, y


@pytest.fixture()
def data_stratified():
    X = dd.from_pandas(
        pd.DataFrame(np.arange(40).reshape(10, 4), columns=list("ABCD")), npartitions=1
    )
    y = dd.from_pandas(
        pd.Series([0, 1, 2, 0, 1, 2, 0, 0, 0, 0], name="TARGET"), npartitions=1
    )
    test_ratio = 0.5
    obj = TrainTestSplit(test_ratio=test_ratio, strategy="stratified", random_state=0)
    return obj, X, y


def test_ordered(data_ordered):
    (
        obj,
        X,
        y,
        X_train_expected,
        X_test_expected,
        y_train_expected,
        y_test_expected,
    ) = data_ordered
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert_frame_equal(X_train.compute(), X_train_expected)
    assert_frame_equal(X_test.compute(), X_test_expected)
    assert_series_equal(y_train.compute(), y_train_expected)
    assert_series_equal(y_test.compute(), y_test_expected)


def test_random(data_random):
    obj, X, y = data_random
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert X_train.compute().shape == (3, 5)
    assert X_test.compute().shape == (5, 5)
    assert y_train.compute().shape == (3,)
    assert y_test.compute().shape == (5,)
    assert (
        len(set(X_train.index.compute().tolist() + X_test.index.compute().tolist()))
        == 8
    )


def test_stratified(data_stratified):
    obj, X, y = data_stratified
    X_train, X_test, y_train, y_test = obj.transform(X, y)
    assert X_train.compute().shape == (5, 4)
    assert X_test.compute().shape == (5, 4)
    assert y_train.compute().shape == (5,)
    assert y_test.compute().shape == (5,)
    assert (y_train == 0).compute().sum() == 3
    assert (y_train == 1).compute().sum() == 1
    assert (y_test == 0).compute().sum() == 3
    assert (y_test == 1).compute().sum() == 1
    assert (
        len(set(X_train.index.compute().tolist() + X_test.index.compute().tolist()))
        == 10
    )
