# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicDayOfWeek


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-05-04T00", None, None],
            "B": ["2020-05-06T06", None, None],
            "C": ["2020-05-08T23", None, None],
            "D": ["2020-05-09T06", None, None],
            "E": ["2020-05-10T06", None, None],
            "X": ["x", None, None],
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
            "A__day_of_week_cos": [1.0, np.nan, np.nan],
            "A__day_of_week_sin": [0.0, np.nan, np.nan],
            "B__day_of_week_cos": [-0.4999999999999998, np.nan, np.nan],
            "B__day_of_week_sin": [0.8660254037844388, np.nan, np.nan],
            "C__day_of_week_cos": [-0.5000000000000004, np.nan, np.nan],
            "C__day_of_week_sin": [-0.8660254037844384, np.nan, np.nan],
            "D__day_of_week_cos": [0.4999999999999993, np.nan, np.nan],
            "D__day_of_week_sin": [-0.866025403784439, np.nan, np.nan],
            "E__day_of_week_cos": [1.0, None, np.nan],
            "E__day_of_week_sin": [-2.4492935982947064e-16, None, np.nan],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = CyclicDayOfWeek(columns=columns).fit(X)
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
        _ = CyclicDayOfWeek(columns=0)
    with pytest.raises(ValueError):
        _ = CyclicDayOfWeek(columns=[])
