# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicDayOfMonth

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
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
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = CyclicDayOfMonth(columns=columns).fit(X)
    return obj, X, X_expected, X_np, X_expected_np


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected, X_np, X_expected_np = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected, X_np, X_expected_np = data_ks
    X_numpy_new = obj.transform_numpy(X_np)
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected_np)
    assert_frame_equal(X_new, X_expected)
