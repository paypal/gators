import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_datatype_columns import DropDatatypeColumns


@pytest.fixture
def data_ks():
    X = ks.DataFrame({"A": [1, 2], "B": [1.0, 2.0], "C": ["q", "w"]})
    obj = DropDatatypeColumns(dtype=float).fit(X)
    X_expected = pd.DataFrame({"A": [1, 2], "C": ["q", "w"]})
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
    assert_frame_equal(X_new, X_expected.astype(object))
