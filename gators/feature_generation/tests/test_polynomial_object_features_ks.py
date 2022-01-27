# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.polynomial_object_features import (
    PolynomialObjectFeatures,
)

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {"A": [None, "b", "c"], "B": ["z", "a", "a"], "C": ["c", "d", "d"]}
    )
    X_expected = pd.DataFrame(
        {
            "A": [None, "b", "c"],
            "B": ["z", "a", "a"],
            "C": ["c", "d", "d"],
            "A__B": ["z", "ba", "ca"],
            "A__C": ["c", "bd", "cd"],
            "B__C": ["zc", "ad", "ad"],
            "A__B__C": ["zc", "bad", "cad"],
        }
    )
    obj = PolynomialObjectFeatures(columns=["A", "B", "C"], degree=3).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas().iloc[:, -3:], X_expected.iloc[:, -3:])


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
