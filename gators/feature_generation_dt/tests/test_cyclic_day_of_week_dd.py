# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import CyclicDayOfWeek


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["2020-05-04T00", None, None],
                "B": ["2020-05-06T06", None, None],
                "C": ["2020-05-08T23", None, None],
                "D": ["2020-05-09T06", None, None],
                "E": ["2020-05-10T06", None, None],
                "X": ["x", None, None],
            }
        ),
        npartitions=1,
    )
    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X_np = X.compute().to_numpy()

    X_expected = pd.DataFrame(
        {
            "A__day_of_week_cos": [1.0, np.nan, np.nan],
            "A__day_of_week_sin": [0.0, np.nan, np.nan],
            "B__day_of_week_cos": [-0.22252093395631434, np.nan, np.nan],
            "B__day_of_week_sin": [0.9749279121818236, np.nan, np.nan],
            "C__day_of_week_cos": [-0.9009688679024191, np.nan, np.nan],
            "C__day_of_week_sin": [-0.433883739117558, np.nan, np.nan],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.compute().copy(), X_expected], axis=1)
    columns = ["A", "B", "C"]
    obj = CyclicDayOfWeek(columns=columns).fit(X)
    return obj, X, X_expected, X_np, X_expected_np


def test_dd(data):
    obj, X, X_expected, X_np, X_expected_np = data
    X_new = obj.transform(X).compute()
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, X_expected, X_np, X_expected_np = data
    X_numpy_new = obj.transform_numpy(X_np)
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected_np)
    assert_frame_equal(X_new, X_expected)
