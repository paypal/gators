# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicMonthOfYear

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["2020-01-01T00", pd.NaT, None],
            "B": ["2020-04-08T06", pd.NaT, None],
            "C": ["2020-07-16T12", pd.NaT, None],
            "D": ["2020-10-24T18", pd.NaT, None],
            "E": ["2020-12-31T23", pd.NaT, None],
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
            "A__month_of_year_cos": [1.0, np.nan, np.nan],
            "A__month_of_year_sin": [0.0, np.nan, np.nan],
            "B__month_of_year_cos": [-0.142314838273285, np.nan, np.nan],
            "B__month_of_year_sin": [0.9898214418809328, np.nan, np.nan],
            "C__month_of_year_cos": [-0.9594929736144975, np.nan, np.nan],
            "C__month_of_year_sin": [-0.28173255684142945, np.nan, np.nan],
            "D__month_of_year_cos": [0.41541501300188605, np.nan, np.nan],
            "D__month_of_year_sin": [-0.9096319953545186, np.nan, np.nan],
            "E__month_of_year_cos": [1.0, np.nan, np.nan],
            "E__month_of_year_sin": [-1.133107779529596e-16, np.nan, np.nan],
        }
    )
    X_expected = pd.concat([X.copy(), X_expected], axis=1)
    obj = CyclicMonthOfYear(columns=columns).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["2020-01-01T00", pd.NaT, None],
            "B": ["2020-04-08T06", pd.NaT, None],
            "C": ["2020-07-16T12", pd.NaT, None],
            "D": ["2020-10-24T18", pd.NaT, None],
            "E": ["2020-12-31T23", pd.NaT, None],
            "X": ["x", "x", "x"],
        }
    )
    columns = ["A", "B", "C", "D", "E"]
    X[columns] = X[columns].astype("datetime64[ns]")
    X_expected = pd.DataFrame(
        {
            "A__month_of_year_cos": [1.0, np.nan, np.nan],
            "A__month_of_year_sin": [0.0, np.nan, np.nan],
            "B__month_of_year_cos": [-0.142314838273285, np.nan, np.nan],
            "B__month_of_year_sin": [0.9898214418809328, np.nan, np.nan],
            "C__month_of_year_cos": [-0.9594929736144975, np.nan, np.nan],
            "C__month_of_year_sin": [-0.28173255684142945, np.nan, np.nan],
            "D__month_of_year_cos": [0.41541501300188605, np.nan, np.nan],
            "D__month_of_year_sin": [-0.9096319953545186, np.nan, np.nan],
            "E__month_of_year_cos": [1.0, np.nan, np.nan],
            "E__month_of_year_sin": [-1.133107779529596e-16, np.nan, np.nan],
        }
    )
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = CyclicMonthOfYear(columns=columns).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.fit_transform(X)
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
        _ = CyclicMonthOfYear(columns=0)
    with pytest.raises(ValueError):
        _ = CyclicMonthOfYear(columns=[])
