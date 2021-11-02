import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.drop_columns import DropColumns


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame({"A": [1, 2], "B": [1.0, 2.0], "C": ["q", "w"]}), npartitions=1
    )
    obj = DropColumns(["A"]).fit(X)
    X_expected = pd.DataFrame({"B": [1.0, 2.0], "C": ["q", "w"]})
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
