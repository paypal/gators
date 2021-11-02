# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning import ConvertColumnDatatype


@pytest.fixture
def data():
    X = pd.DataFrame(
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
def data_obj():
    X = pd.DataFrame(
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


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, dtype=X_numpy_new.dtype
    )
    X_expected = pd.DataFrame(X_expected, columns=X_expected.columns, dtype=float)
    assert_frame_equal(X_new, X_expected)


def test_obj(data_obj):
    obj, X, X_expected = data_obj
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_obj_pd_np(data_obj):
    obj, X, X_expected = data_obj
    X_numpy = X.to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    assert X_numpy_new.tolist() == X.to_numpy().tolist()


def test_input():
    with pytest.raises(TypeError):
        _ = ConvertColumnDatatype(columns=0, datatype=float)
    with pytest.raises(ValueError):
        _ = ConvertColumnDatatype(columns=[], datatype=float)
    with pytest.raises(TypeError):
        _ = ConvertColumnDatatype(columns=["A"], datatype=0)
    with pytest.raises(TypeError):
        _ = ConvertColumnDatatype(columns=["A"], datatype="x")
