# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt.delta_time import DeltaTime

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": {0: "2020-05-04 00:00:00", 1: pd.NaT},
            "B": {0: "2020-05-04 06:00:00", 1: pd.NaT},
            "C": {0: "2020-05-04 12:00:00", 1: pd.NaT},
            "D": {0: "2020-05-04 18:00:00", 1: pd.NaT},
            "E": {0: "2020-05-04 23:00:00", 1: pd.NaT},
            "X": {0: "x", 1: "x"},
        }
    )
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[h]")

    X_expected = pd.DataFrame(
        {
            "B__A__Deltatime[s]": {0: 21600.0, 1: np.nan},
            "C__A__Deltatime[s]": {0: 43200.0, 1: np.nan},
            "D__A__Deltatime[s]": {0: 64800.0, 1: np.nan},
            "E__A__Deltatime[s]": {0: 82800.0, 1: np.nan},
        }
    )
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=["A", "A", "A", "A"]).fit(
        X
    )
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": {0: "2020-05-04 00:00:00", 1: pd.NaT},
            "B": {0: "2020-05-04 06:00:00", 1: pd.NaT},
            "C": {0: "2020-05-04 12:00:00", 1: pd.NaT},
            "D": {0: "2020-05-04 18:00:00", 1: pd.NaT},
            "E": {0: "2020-05-04 23:00:00", 1: pd.NaT},
            "X": {0: "x", 1: "x"},
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X[columns] = X[columns].astype("datetime64[ns]")
    X_expected = pd.DataFrame(
        {
            "B__A__Deltatime[s]": {0: 21600.0, 1: np.nan},
            "C__A__Deltatime[s]": {0: 43200.0, 1: np.nan},
            "D__A__Deltatime[s]": {0: 64800.0, 1: np.nan},
            "E__A__Deltatime[s]": {0: 82800.0, 1: np.nan},
        }
    )
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=["A", "A", "A", "A"]).fit(
        X
    )
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
        _ = DeltaTime(columns_a=0, columns_b=["A", "A", "A", "A"])
    with pytest.raises(TypeError):
        _ = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=0)
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=[], columns_b=["A", "A", "A", "A"])
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=["B", "C", "D", "E"], columns_b=[])
    with pytest.raises(ValueError):
        _ = DeltaTime(columns_a=["B"], columns_b=["A", "A", "A", "A"])


def test_init_fit(data):
    _, _, X = data
    with pytest.raises(TypeError):
        _ = DeltaTime(
            columns_a=["B", "C", "D", "X"], columns_b=["A", "A", "A", "A"]
        ).fit(X)
