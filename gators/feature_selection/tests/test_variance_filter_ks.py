# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_selection.variance_filter import VarianceFilter

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    min_var = 2.0
    X = ks.DataFrame(
        {
            "A": [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [1, 2, 3, 1, 2, 4],
        }
    )
    X_expected = X[["A", "D"]].copy().to_pandas()
    y = ks.Series([1, 1, 1, 0, 0, 0])
    obj = VarianceFilter(min_var=min_var).fit(X, y)
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
