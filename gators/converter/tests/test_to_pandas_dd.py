# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.converter.to_pandas import ToPandas


@pytest.fixture
def data_dd():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "q": [0.0, 3.0, 6.0],
                "w": [1.0, 4.0, 7.0],
                "e": [2.0, 5.0, 8.0],
            }
        ),
        npartitions=1,
    )
    y = dd.from_pandas(pd.Series([0, 0, 1], name="TARGET"), npartitions=1)
    return X, y, X.compute(), y.compute()


def test_dd(data_dd):
    X, y, X_expected, y_expected = data_dd
    X_new, y_new = ToPandas().transform(X, y)
    assert_frame_equal(X_new, X_expected)
    assert_series_equal(y_new, y_expected)
