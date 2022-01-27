# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning.binning import Binning
from gators.feature_selection.information_value import InformationValue

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    k = 3
    X = ks.DataFrame(
        {
            "A": ["a", "b", "a", "b", "c", "b"],
            "B": ["true", "true", "false", "true", "false", "false"],
            "D": ["a", "b", "c", "d", "e", "f"],
            "F": ["e", "f", "g", "e", "f", "g"],
        }
    )
    X_expected = X[["A", "B", "F"]].to_pandas().copy()
    y = ks.Series([1, 1, 1, 0, 0, 0], name="TARGET")
    obj = InformationValue(k=k).fit(X, y)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)
