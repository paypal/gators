# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_selection.supervized_correlation_filter import (
    SupervizedCorrelationFilter,
)


@pytest.fixture
def data():
    max_corr = 0.8
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = X[["A", "B", "D"]]
    feature_importances = pd.Series({"A": 1, "B": 0.8, "D": 0.7, "F": 0.1})
    obj = SupervizedCorrelationFilter(
        feature_importances=feature_importances, max_corr=max_corr
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_():
    max_corr = 0.8
    X = pd.DataFrame(
        {
            "D": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "A": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "B": [1, 1, 0, 1, 0, 0],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = X[["A", "B", "F"]]
    feature_importances = pd.Series({"A": 1, "B": 0.7, "D": 0.1, "F": 0.1})
    obj = SupervizedCorrelationFilter(
        feature_importances=feature_importances, max_corr=max_corr
    ).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(np.float64))


def test_pd(data_):
    obj, X, X_expected = data_
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data_):
    obj, X, X_expected = data_
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(np.float64))


def test_init():
    with pytest.raises(TypeError):
        _ = SupervizedCorrelationFilter(feature_importances="a", max_corr="a")
    with pytest.raises(TypeError):
        _ = SupervizedCorrelationFilter(
            feature_importances=pd.Series({"A": 0.1}), max_corr="a"
        )
