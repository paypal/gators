# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation_dt import OrdinalMinuteOfHour

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": ["2020-05-04 00:00:00", None],
            "B": ["2020-05-06 00:10:00", None],
            "C": ["2020-05-08 00:20:00", None],
            "D": ["2020-05-09 00:40:00", None],
            "E": ["2020-05-09 00:59:00", None],
            "X": ["x", "x"],
        }
    )
    X_np = X.to_numpy()
    columns = ["A", "B", "C", "D", "E"]
    X[columns] = X[columns].astype("datetime64[ns]")
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
    X_expected = pd.concat([X.to_pandas().copy(), X_expected], axis=1)
    obj = OrdinalMinuteOfHour(columns=columns).fit(X)
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
