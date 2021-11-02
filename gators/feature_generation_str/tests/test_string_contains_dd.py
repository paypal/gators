# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import StringContains


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
    obj = StringContains(columns=list("DEF"), contains_vec=["1", "2", "0"]).fit(X)
    columns_expected = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "D__contains_1",
        "E__contains_2",
        "F__contains_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", 1.0, 1.0, 0.0],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


@pytest.fixture
def data_with_names():
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
    obj = StringContains(
        columns=list("DEF"),
        contains_vec=["1", "2", "0"],
        column_names=["D_with_1", "E_with_2", "F_with_0"],
    ).fit(X)
    columns_expected = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "D_with_1",
        "E_with_2",
        "F_with_0",
    ]
    X_expected = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, "0", "0", "0", 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, "1Q", "W2", "Q", 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, "1QQ", "W2W", "", 1.0, 1.0, 0.0],
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


def test_names_dd(data_with_names):
    obj, X, X_expected = data_with_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_names_dd_np(data_with_names):
    obj, X, X_expected = data_with_names
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
