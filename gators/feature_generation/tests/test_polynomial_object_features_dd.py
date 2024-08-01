# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.polynomial_object_features import (
    PolynomialObjectFeatures,
)


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {"A": [None, "b", "c"], "B": ["z", "a", "a"], "C": ["c", "d", "d"]}
        ),
        npartitions=1,
    )
    X_np = pd.DataFrame(
        {"A": [None, "b", "c"], "B": ["z", "a", "a"], "C": ["c", "d", "d"]}
    ).to_numpy()
    X_expected = pd.DataFrame(
        {
            "A": [None, "b", "c"],
            "B": ["z", "a", "a"],
            "C": ["c", "d", "d"],
            "A__x__B": ["z", "ba", "ca"],
            "A__x__C": ["c", "bd", "cd"],
            "B__x__C": ["zc", "ad", "ad"],
            "A__x__B__x__C": ["zc", "bad", "cad"],
        }
    )
    obj = PolynomialObjectFeatures(columns=["A", "B", "C"], degree=3).fit(X)
    return obj, X, X_np, X_expected


def test_dd(data):
    obj, X, _, X_expected = data
    X_new = obj.transform(X).compute()
    X_new = X_new.astype(object)
    assert_frame_equal(X_new.iloc[:, -3:], X_expected.iloc[:, -3:])


def test_dd_np(data):
    obj, X, X_np, X_expected = data
    X_numpy_new = obj.transform_numpy(X_np)
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)
