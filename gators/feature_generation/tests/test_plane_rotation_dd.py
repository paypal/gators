# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.plane_rotation import PlaneRotation


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame({"X": [200.0, 210.0], "Y": [140.0, 160.0], "Z": [100.0, 125.0]}),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "X": [200.0, 210.0],
            "Y": [140.0, 160.0],
            "Z": [100.0, 125.0],
            "XY_x_45deg": [42.42640687119287, 35.35533905932739],
            "XY_y_45deg": [240.41630560342614, 261.62950903902254],
            "XY_x_60deg": [-21.243556529821376, -33.56406460551014],
            "XY_y_60deg": [243.20508075688775, 261.8653347947321],
            "XZ_x_45deg": [70.71067811865477, 60.104076400856556],
            "XZ_y_45deg": [212.13203435596424, 236.8807716974934],
            "XZ_x_60deg": [13.397459621556166, -3.253175473054796],
            "XZ_y_60deg": [223.20508075688775, 244.36533479473212],
        }
    )
    obj = PlaneRotation(columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60]).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_object():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "X": [200.0, 210.0],
                "A": ["a", "b"],
                "Y": [140.0, 160.0],
                "Z": [100.0, 125.0],
            }
        ),
        npartitions=1,
    )
    X_expected = pd.DataFrame(
        {
            "X": [200.0, 210.0],
            "A": ["a", "b"],
            "Y": [140.0, 160.0],
            "Z": [100.0, 125.0],
            "XY_x_45deg": [42.42640687119287, 35.35533905932739],
            "XY_y_45deg": [240.41630560342614, 261.62950903902254],
            "XY_x_60deg": [-21.243556529821376, -33.56406460551014],
            "XY_y_60deg": [243.20508075688775, 261.8653347947321],
            "XZ_x_45deg": [70.71067811865477, 60.104076400856556],
            "XZ_y_45deg": [212.13203435596424, 236.8807716974934],
            "XZ_x_60deg": [13.397459621556166, -3.253175473054796],
            "XZ_y_60deg": [223.20508075688775, 244.36533479473212],
        }
    )
    obj = PlaneRotation(columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60]).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected)


def test_object_dd(data_object):
    obj, X, X_expected = data_object
    X_new = obj.transform(X)
    assert_frame_equal(X_new.compute(), X_expected)


def test_object_dd_np(data_object):
    obj, X, X_expected = data_object
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert_frame_equal(X_new, pd.DataFrame(X_expected.to_numpy()))
