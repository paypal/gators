# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicHourOfDay

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-01-01T00", None, np.nan],
            "B": ["2020-04-08T06", None, np.nan],
            "C": ["2020-07-16T12", None, np.nan],
            "D": ["2020-10-24T18", None, np.nan],
            "E": ["2020-12-31T23", None, np.nan],
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
            "A__hour_of_day_cos": [1.0, None, np.nan],
            "A__hour_of_day_sin": [0.0, None, np.nan],
            "B__hour_of_day_cos": [-0.06824241336467089, None, np.nan],
            "B__hour_of_day_sin": [0.9976687691905392, None, np.nan],
            "C__hour_of_day_cos": [-0.9906859460363308, None, np.nan],
            "C__hour_of_day_sin": [-0.13616664909624643, None, np.nan],
            "D__hour_of_day_cos": [0.20345601305263328, None, np.nan],
            "D__hour_of_day_sin": [-0.979084087682323, None, np.nan],
            "E__hour_of_day_cos": [1.0, None, np.nan],
            "E__hour_of_day_sin": [-2.4492935982947064e-16, None, np.nan],
        }
    )
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = CyclicHourOfDay(columns=columns).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["2020-01-01T00", None, np.nan],
            "B": ["2020-04-08T06", None, np.nan],
            "C": ["2020-07-16T12", None, np.nan],
            "D": ["2020-10-24T18", None, np.nan],
            "E": ["2020-12-31T23", None, np.nan],
            "X": ["x", "x", "x"],
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X[columns] = X[columns].astype("datetime64[ns]")
    X_expected = pd.DataFrame(
        {
            "A__hour_of_day_cos": [1.0, None, np.nan],
            "A__hour_of_day_sin": [0.0, None, np.nan],
            "B__hour_of_day_cos": [-0.06824241336467089, None, np.nan],
            "B__hour_of_day_sin": [0.9976687691905392, None, np.nan],
            "C__hour_of_day_cos": [-0.9906859460363308, None, np.nan],
            "C__hour_of_day_sin": [-0.13616664909624643, None, np.nan],
            "D__hour_of_day_cos": [0.20345601305263328, None, np.nan],
            "D__hour_of_day_sin": [-0.979084087682323, None, np.nan],
            "E__hour_of_day_cos": [1.0, None, np.nan],
            "E__hour_of_day_sin": [-2.4492935982947064e-16, None, np.nan],
        }
    )
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = CyclicHourOfDay(columns=columns).fit(X)
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
        _ = CyclicHourOfDay(columns=0)
    with pytest.raises(ValueError):
        _ = CyclicHourOfDay(columns=[])
