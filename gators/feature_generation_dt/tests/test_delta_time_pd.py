# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt.delta_time import DeltaTime


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-05-04 00:00:00", None],
            "B": ["2020-05-04 06:00:00", None],
            "C": ["2020-05-04 12:00:00", None],
            "D": ["2020-05-04 18:00:00", None],
            "E": ["2020-05-04 23:00:00", None],
            "X": ["x", "x"],
        }
    )
    X_np = X.to_numpy()
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[h]")
    X_expected = pd.DataFrame(
        {
            "B__A__Deltatime[s]": [21600.0, np.nan],
            "C__A__Deltatime[s]": [43200.0, np.nan],
            "D__A__Deltatime[s]": [64800.0, np.nan],
            "E__A__Deltatime[s]": [82800.0, np.nan],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=["A", "A", "A", "A"]).fit(
        X
    )
    return obj, X, X_expected, X_np, X_expected_np


def test_pd(data):
    obj, X, X_expected, X_np, X_expected_np = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected, X_np, X_expected_np = data
    X_numpy_new = obj.transform_numpy(X_np)
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected_np)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = DeltaTime(columns_a=0, columns_b=["A", "A", "A", "A"])
    with pytest.raises(TypeError):
        _ = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=0)
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=[], columns_b=["A", "A", "A", "A"])
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=[])
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=["B"], columns_b=["A", "A", "A", "A"])


def test_init_fit():
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=["B", "C", "D"], columns_b=["A", "A", "A", "A"])


def test_fit():
    X = pd.DataFrame({"A": [0], "B": [0]})
    obj = DeltaTime(columns_a=["A"], columns_b=["B"])
    with pytest.raises(TypeError):
        _ = obj.fit(X)
