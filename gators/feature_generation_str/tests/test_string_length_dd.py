# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import StringLength


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0],
                "C": [0.0, 0.0, 0.0],
                "D": ["Q", "QQ", "QQQ"],
                "E": ["W", "WW", "WWW"],
                "F": ["nan", "", ""],
            }
        ),
        npartitions=1,
    )
    obj = StringLength(columns=list("DEF")).fit(X)
    columns_expected = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "D__length",
        "E__length",
        "F__length",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "Q", "W", "nan", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "QQ", "WW", "", 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, "QQQ", "WWW", "", 3.0, 3.0, 0.0],
        ],
        columns=columns_expected,
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
