# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicHourOfDay


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-01-01T00", None, None],
            "B": ["2020-04-08T06", None, None],
            "C": ["2020-07-16T12", None, None],
            "D": ["2020-10-24T18", None, None],
            "E": ["2020-12-31T23", None, None],
            "X": ["x", "x", "x"],
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
            "A__hour_of_day_cos": [1.0, None, np.nan],
            "A__hour_of_day_sin": [0.0, None, np.nan],
            "B__hour_of_day_cos": [0.0, None, np.nan],
            "B__hour_of_day_sin": [1.0, None, np.nan],
            "C__hour_of_day_cos": [-1, None, np.nan],
            "C__hour_of_day_sin": [0.0, None, np.nan],
            "D__hour_of_day_cos": [0.0, None, np.nan],
            "D__hour_of_day_sin": [-1.0, None, np.nan],
            "E__hour_of_day_cos": [0.9659258262890681, None, np.nan],
            "E__hour_of_day_sin": [-0.2588190451025215, None, np.nan],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = CyclicHourOfDay(columns=columns).fit(X)
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
        _ = CyclicHourOfDay(columns=0)
    with pytest.raises(ValueError):
        _ = CyclicHourOfDay(columns=[])
