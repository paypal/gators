# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import UpperCase


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0],
                "C": [0.0, 0.0, 0.0],
                "D": ["q", "qq", "QQq"],
                "E": ["w", "WW", "WWw"],
                "F": ["abc", "", ""],
            }
        ),
        npartitions=1,
    )

    obj = UpperCase(columns=list("DEF")).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": ["Q", "QQ", "QQQ"],
            "E": ["W", "WW", "WWW"],
            "F": ["ABC", "", ""],
        }
    )
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute()
    X_new[["D", "E", "F"]] = X_new[["D", "E", "F"]].astype(object)
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
