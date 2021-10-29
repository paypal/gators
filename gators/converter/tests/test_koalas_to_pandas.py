import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gators.converter.koalas_to_pandas import KoalasToPandas

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "q": {0: 0.0, 1: 3.0, 2: 6.0},
            "w": {0: 1.0, 1: 4.0, 2: 7.0},
            "e": {0: 2.0, 1: 5.0, 2: 8.0},
        }
    )
    y = ks.Series([0, 0, 1], name="TARGET")
    return X, y, X.to_pandas(), y.to_pandas()


@pytest.mark.koalas
def test_ks(data_ks):
    X_ks, y_ks, X_expected, y_expected = data_ks
    X_new, y_new = KoalasToPandas().transform(X_ks, y_ks)
    assert_frame_equal(X_new, X_expected)
    assert_series_equal(y_new, y_expected)


def test_input():
    with pytest.raises(TypeError):
        _ = KoalasToPandas().transform(pd.DataFrame(), pd.DataFrame())
