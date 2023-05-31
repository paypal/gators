# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicDayOfMonth


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2021-02-28T06", None, None],
            "B": ["2020-02-29T06", None, None],
            "C": ["2020-03-01T12", None, None],
            "D": ["2020-04-01T18", None, None],
            "E": ["2020-05-31T23", None, None],
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
            "A__day_of_month_cos": [0.9749279121818235, np.nan, np.nan],
            "A__day_of_month_sin": [-0.22252093395631464, np.nan, np.nan],
            "B__day_of_month_cos": [0.9766205557100867, np.nan, np.nan],
            "B__day_of_month_sin": [-0.2149704402110244, np.nan, np.nan],
            "C__day_of_month_cos": [1.0, np.nan, np.nan],
            "C__day_of_month_sin": [0.0, np.nan, np.nan],
            "D__day_of_month_cos": [1.0, np.nan, np.nan],
            "D__day_of_month_sin": [0.0, np.nan, np.nan],
            "E__day_of_month_cos": [0.9795299412524945, np.nan, np.nan],
            "E__day_of_month_sin": [-0.20129852008866028, np.nan, np.nan],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = CyclicDayOfMonth(columns=columns).fit(X)
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
        _ = CyclicDayOfMonth(columns=0)
    with pytest.raises(ValueError):
        _ = CyclicDayOfMonth(columns=[])
