# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicDayOfMonth

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2021-02-28T06", None, np.nan],
            "B": ["2020-02-29T06", None, np.nan],
            "C": ["2020-03-01T12", None, np.nan],
            "D": ["2020-04-01T18", None, np.nan],
            "E": ["2020-05-31T23", None, np.nan],
            "X": ["x", "x", "x"],
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[h]")
    X_expected = pd.DataFrame(
        {
            "A__day_of_month_cos": [1.0, np.nan, np.nan],
            "A__day_of_month_sin": [0.0, np.nan, np.nan],
            "B__day_of_month_cos": [1.0, np.nan, np.nan],
            "B__day_of_month_sin": [0.0, np.nan, np.nan],
            "C__day_of_month_cos": [1.0, np.nan, np.nan],
            "C__day_of_month_sin": [0.0, np.nan, np.nan],
            "D__day_of_month_cos": [1.0, np.nan, np.nan],
            "D__day_of_month_sin": [0.0, np.nan, np.nan],
            "E__day_of_month_cos": [1.0, np.nan, np.nan],
            "E__day_of_month_sin": [0.0, np.nan, np.nan],
        }
    )
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = CyclicDayOfMonth(columns=columns).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["2021-02-28T06", None, np.nan],
            "B": ["2020-02-29T06", None, np.nan],
            "C": ["2020-03-01T12", None, np.nan],
            "D": ["2020-04-01T18", None, np.nan],
            "E": ["2020-05-31T23", None, np.nan],
            "X": ["x", "x", "x"],
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X[columns] = X[columns].astype("datetime64[ns]")
    X_expected = pd.DataFrame(
        {
            "A__day_of_month_cos": [1.0, np.nan, np.nan],
            "A__day_of_month_sin": [0.0, np.nan, np.nan],
            "B__day_of_month_cos": [1.0, np.nan, np.nan],
            "B__day_of_month_sin": [0.0, np.nan, np.nan],
            "C__day_of_month_cos": [1.0, np.nan, np.nan],
            "C__day_of_month_sin": [0.0, np.nan, np.nan],
            "D__day_of_month_cos": [1.0, np.nan, np.nan],
            "D__day_of_month_sin": [0.0, np.nan, np.nan],
            "E__day_of_month_cos": [1.0, np.nan, np.nan],
            "E__day_of_month_sin": [0.0, np.nan, np.nan],
        }
    )
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = CyclicDayOfMonth(columns=columns).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_init():
    with pytest.raises(TypeError):
        _ = CyclicDayOfMonth(columns=0)
    with pytest.raises(ValueError):
        _ = CyclicDayOfMonth(columns=[])
