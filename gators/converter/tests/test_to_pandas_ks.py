# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.converter.to_pandas import ToPandas

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "q": [0.0, 3.0, 6.0],
            "w": [1.0, 4.0, 7.0],
            "e": [2.0, 5.0, 8.0],
        }
    )
    y = ks.Series([0, 0, 1], name="TARGET")
    return X, y, X.to_pandas(), y.to_pandas()


@pytest.mark.koalas
def test_ks(data_ks):
    X_ks, y_ks, X_expected, y_expected = data_ks
    X_new, y_new = ToPandas().transform(X_ks, y_ks)
    assert_frame_equal(X_new, X_expected)
    assert_series_equal(y_new, y_expected)
