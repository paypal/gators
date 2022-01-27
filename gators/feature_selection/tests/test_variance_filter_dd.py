# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_selection.variance_filter import VarianceFilter


@pytest.fixture
def data():
    min_var = 2.0
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
                "B": [1, 1, 0, 1, 0, 0],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [1, 2, 3, 1, 2, 4],
            }
        ),
        npartitions=1,
    )
    X_expected = X[["A", "D"]].copy().compute()
    y = dd.from_pandas(pd.Series([1, 1, 1, 0, 0, 0]), npartitions=1)
    obj = VarianceFilter(min_var=min_var).fit(X, y)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_empty_drop_columns(data):
    obj, X, _ = data
    obj.columns_to_drop = []
    assert_frame_equal(obj.transform(X).compute(), X.compute())
