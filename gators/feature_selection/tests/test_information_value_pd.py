# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.binning import Binning
from gators.feature_selection.information_value import InformationValue


@pytest.fixture
def data():
    k = 3
    X = pd.DataFrame(
        {
            "A": ["a", "b", "a", "b", "c", "b"],
            "B": ["true", "true", "false", "true", "false", "false"],
            "D": ["a", "b", "c", "d", "e", "f"],
            "F": ["e", "f", "g", "e", "f", "g"],
        }
    )
    X_expected = X[["A", "B", "F"]].copy()
    y = pd.Series([1, 1, 1, 0, 0, 0], name="TARGET")
    obj = InformationValue(k=k).fit(X, y)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = InformationValue(k="a")
    with pytest.raises(TypeError):
        _ = InformationValue(k=2, regularization="a")
    with pytest.raises(TypeError):
        _ = InformationValue(k=2, max_iv="a")
