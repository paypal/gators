# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import Extract


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0],
                "C": [0.0, 0.0, 0.0],
                "D": ["0", "1Q", "1QQ"],
                "E": ["0", "W2", "W2W"],
                "F": ["0", "Q", ""],
            }
        ),
        npartitions=1,
    )
    obj = Extract(columns=list("DEF"), i_min_vec=[0, 1, 2], i_max_vec=[1, 2, 3]).fit(X)
    columns_expected = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "D__substring_0_to_1",
        "E__substring_1_to_2",
        "F__substring_2_to_3",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", "0", "", ""],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", "1", "2", ""],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", "1", "2", ""],
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
