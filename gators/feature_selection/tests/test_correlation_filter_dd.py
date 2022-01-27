# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_selection.correlation_filter import CorrelationFilter


@pytest.fixture
def data():
    max_corr = 0.8
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
                "B": [1, 1, 0, 1, 0, 0],
                "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
                "F": [3, 1, 2, 1, 2, 3],
            }
        ),
        npartitions=1,
    )
    X_expected = X[["B", "D", "F"]].compute().copy()
    obj = CorrelationFilter(max_corr=max_corr).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(np.float64))
