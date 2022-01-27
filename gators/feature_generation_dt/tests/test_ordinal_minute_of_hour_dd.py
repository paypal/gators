# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import OrdinalMinuteOfHour


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["2020-05-04 00:00:00", None],
                "B": ["2020-05-06 00:10:00", None],
                "C": ["2020-05-08 00:20:00", None],
                "D": ["2020-05-09 00:40:00", None],
                "E": ["2020-05-09 00:59:00", None],
                "X": ["x", "x"],
            }
        ),
        npartitions=1,
    )
    X_np = X.compute().to_numpy()

    X["A"] = X["A"].astype("datetime64[ns]")
    X["B"] = X["B"].astype("datetime64[ms]")
    X["C"] = X["C"].astype("datetime64[s]")
    X["D"] = X["D"].astype("datetime64[m]")
    X["E"] = X["E"].astype("datetime64[m]")
    X_expected = pd.DataFrame(
        {
            "A__minute_of_hour": [0.0, np.nan],
            "B__minute_of_hour": [10.0, np.nan],
            "C__minute_of_hour": [20.0, np.nan],
            "D__minute_of_hour": [40.0, np.nan],
            "E__minute_of_hour": [59.0, np.nan],
        }
    )
    X_expected_np = np.concatenate((X_np, X_expected.to_numpy()), axis=1)
    X_expected = pd.concat([X.compute().copy(), X_expected], axis=1)
    columns = ["A", "B", "C", "D", "E"]
    obj = OrdinalMinuteOfHour(columns=columns).fit(X)
    return obj, X, X_expected, X_np, X_expected_np


def test_dd(data):
    obj, X, X_expected, X_np, X_expected_np = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected, X_np, X_expected_np = data
    X_numpy_new = obj.transform_numpy(X_np)
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected_np)
    assert_frame_equal(X_new, X_expected)
