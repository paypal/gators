# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.is_null import IsNull

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": [np.nan, 3.0, 6.0],
            "B": [np.nan, 4.0, 7.0],
            "C": [np.nan, 5.0, 8.0],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [np.nan, 3.0, 6.0],
            "B": [np.nan, 4.0, 7.0],
            "C": [np.nan, 5.0, 8.0],
            "A__is_null": [1.0, 0.0, 0.0],
            "B__is_null": [1.0, 0.0, 0.0],
            "C__is_null": [1.0, 0.0, 0.0],
        }
    )
    obj = IsNull(columns=list("ABC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names_ks():
    X = ks.DataFrame(
        {
            "A": [np.nan, 3.0, 6.0],
            "B": [np.nan, 4.0, 7.0],
            "C": [np.nan, 5.0, 8.0],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [np.nan, 3.0, 6.0],
            "B": [np.nan, 4.0, 7.0],
            "C": [np.nan, 5.0, 8.0],
            "AIsNull": [1.0, 0.0, 0.0],
            "BIsNull": [1.0, 0.0, 0.0],
            "CIsNull": [1.0, 0.0, 0.0],
        }
    )
    obj = IsNull(
        columns=list("ABC"), column_names=["AIsNull", "BIsNull", "CIsNull"]
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_obj_ks():
    X = ks.DataFrame(
        {
            "A": [None, "a", "b"],
            "B": [None, "c", "d"],
            "C": [None, "e", "f"],
            "D": [0, 1, np.nan],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [None, "a", "b"],
            "B": [None, "c", "d"],
            "C": [None, "e", "f"],
            "D": [0, 1, np.nan],
            "A__is_null": [1.0, 0.0, 0.0],
            "B__is_null": [1.0, 0.0, 0.0],
            "C__is_null": [1.0, 0.0, 0.0],
            "D__is_null": [0.0, 0.0, 1.0],
        }
    )
    obj = IsNull(columns=list("ABCD")).fit(X)
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
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_names_ks_np(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_obj_ks(data_obj_ks):
    obj, X, X_expected = data_obj_ks
    X_new = obj.transform(X).to_pandas()
    assert_frame_equal(
        X_new.iloc[:, 4:].astype(float), X_expected.iloc[:, 4:].astype(float)
    )


@pytest.mark.koalas
def test_obj_ks_np(data_obj_ks):
    obj, X, X_expected = data_obj_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(
        X_new.iloc[:, 4:].astype(float), X_expected.iloc[:, 4:].astype(float)
    )
