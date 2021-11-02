# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_str import StringLength


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0],
            "B": [0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0],
            "D": ["Q", "QQ", "QQQ"],
            "E": ["W", "WW", "WWW"],
            "F": ["nan", None, ""],
        }
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
            [0.0, 0.0, 0.0, "QQ", "WW", None, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, "QQQ", "WWW", "", 3.0, 3.0, 0.0],
        ],
        columns=columns_expected,
    )
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = StringLength(columns="x")
    with pytest.raises(ValueError):
        _ = StringLength(columns=[])
