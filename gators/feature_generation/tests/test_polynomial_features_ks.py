import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.polynomial_features import PolynomialFeatures

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_interaction_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
        }
    )
    obj = PolynomialFeatures(interaction_only=True, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
                [3.0, 4.0, 5.0, 12.0, 15.0, 20.0],
                [6.0, 7.0, 8.0, 42.0, 48.0, 56.0],
            ]
        ),
        columns=["A", "B", "C", "A__x__B", "A__x__C", "B__x__C"],
    )
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "D": ["a", "b", "c"],
        },
    )
    obj = PolynomialFeatures(interaction_only=False, columns=["A", "B", "C"]).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "D": ["a", "b", "c"],
            "A__x__A": [0.0, 9.0, 36.0],
            "A__x__B": [0.0, 12.0, 42.0],
            "A__x__C": [0.0, 15.0, 48.0],
            "B__x__B": [1.0, 16.0, 49.0],
            "B__x__C": [2.0, 20.0, 56.0],
            "C__x__C": [4.0, 25.0, 64.0],
        }
    )
    return obj, X, X_expected


@pytest.mark.koalas
def test_interaction_ks(data_interaction_ks):
    obj, X, X_expected = data_interaction_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_interaction_ks_np(data_interaction_ks):
    obj, X, X_expected = data_interaction_ks
    X_new = obj.transform_numpy(X.to_numpy())
    assert np.allclose(X_new, X_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
