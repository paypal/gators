# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import OrdinalDayOfMonth


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-05-01 00:00:00", None],
            "B": ["2020-05-08 06:00:00", None],
            "C": ["2020-05-16 12:00:00", None],
            "D": ["2020-05-24 18:00:00", None],
            "E": ["2020-05-30 23:00:00", None],
            "X": ["x", "x"],
        }
    )
    X_np = X.to_numpy()
    columns = ["A", "B", "C", "D", "E"]
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[h]")
    X_expected = pd.DataFrame(
        {
            "A__day_of_month": [1.0, None],
            "B__day_of_month": [8.0, None],
            "C__day_of_month": [16.0, None],
            "D__day_of_month": [24.0, None],
            "E__day_of_month": [30.0, None],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = OrdinalDayOfMonth(columns=columns).fit(X)
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
        _ = OrdinalDayOfMonth(columns=0)
    with pytest.raises(ValueError):
        _ = OrdinalDayOfMonth(columns=[])
