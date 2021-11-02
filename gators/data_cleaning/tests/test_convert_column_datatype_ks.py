# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning import ConvertColumnDatatype

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": [True, False, True, False],
            "B": [True, True, True, False],
            "C": [True, True, True, True],
            "D": [1, 2, 3, 4],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [1.0, 0.0, 1.0, 0.0],
            "B": [1.0, 1.0, 1.0, 0.0],
            "C": [1.0, 1.0, 1.0, 1.0],
            "D": [1, 2, 3, 4],
        }
    )
    obj = ConvertColumnDatatype(columns=["A", "B", "C"], datatype=float).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_obj_ks():
    X = ks.DataFrame(
        {
            "A": ["2020-01-01 00:00:00", "2020-04-08 06:00:00"],
            "B": [True, False],
            "C": [True, True],
            "D": [1, 2],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["2020-01-01 00:00:00", "2020-04-08 06:00:00"],
            "B": [True, False],
            "C": [True, True],
            "D": [1, 2],
        }
    )
    X_expected["A"] = X_expected["A"].astype("datetime64[ns]")
    obj = ConvertColumnDatatype(columns=["A"], datatype="datetime64[ns]").fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    assert X_numpy_new.tolist() == X_expected.to_numpy().tolist()


@pytest.mark.koalas
def test_obj_ks(data_obj_ks):
    obj, X, X_expected = data_obj_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_obj_ks_np(data_obj_ks):
    obj, X, X_expected = data_obj_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    assert X_numpy_new.tolist() == X.to_numpy().tolist()
