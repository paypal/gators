# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import BinSingleTargetClassCategories


@pytest.fixture
def data():
    X = ks.DataFrame(
        {
            "A": ["_0", "_1", "_2", "_2", "_1"],
            "B": ["_1", "_2", "_1", "_1", "_1"],
            "C": ["_0", "_0", "_1", "_2", "_2"],
            "D": ["_0", "_0", "_1", "_1", "_1"],
            "E": [1, 2, 3, 4, 5],
        }
    )
    y = ks.Series([0, 1, 1, 0, 0], name="Target")

    X_expected = pd.DataFrame(
        {
            "A": ["_0|_1", "_0|_1", "_2", "_2", "_0|_1"],
            "B": ["_1|_2", "_1|_2", "_1|_2", "_1|_2", "_1|_2"],
            "C": ["_0|_1|_2", "_0|_1|_2", "_0|_1|_2", "_0|_1|_2", "_0|_1|_2"],
            "D": ["_0", "_0", "_1", "_1", "_1"],
            "E": [1, 2, 3, 4, 5],
        }
    )
    obj = BinSingleTargetClassCategories().fit(X, y)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def tesks_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    X_expected.index = X_new.index
    assert_frame_equal(X_new, X_expected.astype(object))
