# License: Apache-2
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders import FrequencyEncoder


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1.0, 2.0, 3.0],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [2.0, 2.0, 1.0],
            "B": [1.0, 2.0, 2.0],
            "C": [2.0, 1.0, 2.0],
            "D": [1.0, 2.0, 3.0],
        }
    )
    obj = FrequencyEncoder().fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = pd.DataFrame(
        np.zeros((3, 3)),
        columns=list("ABC"),
    )
    obj = FrequencyEncoder().fit(X)
    return obj, X, X.copy()


@pytest.fixture
def data_not_inplace():
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1.0, 2.0, 3.0],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1.0, 2.0, 3.0],
            "A__frequency": [2.0, 2.0, 1.0],
            "B__frequency": [1.0, 2.0, 2.0],
            "C__frequency": [2.0, 1.0, 2.0],
        }
    )
    X_expected_numpy = pd.DataFrame(
        {
            "A": ["Q", "Q", "W"],
            "B": ["Q", "W", "W"],
            "C": ["W", "Q", "W"],
            "D": [1.0, 2.0, 3.0],
            "A__frequency": [2.0, 2.0, 1.0],
            "B__frequency": [1.0, 2.0, 2.0],
            "C__frequency": [2.0, 1.0, 2.0],
        }
    ).astype(object)
    obj = FrequencyEncoder(inplace=False).fit(X)
    return obj, X, X_expected, X_expected_numpy


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(float))


def test_no_cat_pd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_pd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_data_not_inplace_pd(data_not_inplace):
    obj, X, X_expected, _ = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_data_not_inplace_pd_np(data_not_inplace):
    obj, X, _, X_expected_numpy = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected_numpy.columns)
    assert_frame_equal(X_new, X_expected_numpy)
