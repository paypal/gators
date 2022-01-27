# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import SplitExtract


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0],
                "C": [0.0, 0.0, 0.0],
                "D": ["0", "1*Q", "1Q*QQ"],
                "E": ["0", "W*2", "W2*WW"],
                "F": ["0", "Q*", "qwert*"],
            }
        ),
        npartitions=1,
    )
    obj = SplitExtract(list("DEF"), list("***"), [1, 1, 0]).fit(X)
    columns_expected = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "D__split_by_*_idx_1",
        "E__split_by_*_idx_1",
        "F__split_by_*_idx_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", "MISSING", "MISSING", "0"],
            [0.0, 0.0, 0.0, "1*Q", "W*2", "Q*", "Q", "2", "Q"],
            [0.0, 0.0, 0.0, "1Q*QQ", "W2*WW", "qwert*", "QQ", "WW", "qwert"],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
