# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.plane_rotation import PlaneRotation

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        [[200.0, 140.0, 100.0], [210.0, 160.0, 125.0]], columns=list("XYZ")
    )
    X_expected = pd.DataFrame(
        {
            "X": {0: 200.0, 1: 210.0},
            "Y": {0: 140.0, 1: 160.0},
            "Z": {0: 100.0, 1: 125.0},
            "XY_x_45deg": {0: 42.42640687119287, 1: 35.35533905932739},
            "XY_y_45deg": {0: 240.41630560342614, 1: 261.62950903902254},
            "XY_x_60deg": {0: -21.243556529821376, 1: -33.56406460551014},
            "XY_y_60deg": {0: 243.20508075688775, 1: 261.8653347947321},
            "XZ_x_45deg": {0: 70.71067811865477, 1: 60.104076400856556},
            "XZ_y_45deg": {0: 212.13203435596424, 1: 236.8807716974934},
            "XZ_x_60deg": {0: 13.397459621556166, 1: -3.253175473054796},
            "XZ_y_60deg": {0: 223.20508075688775, 1: 244.36533479473212},
        }
    )
    obj = PlaneRotation(columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60]).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_float32():
    X = pd.DataFrame([[200, 140, 100], [210, 160, 125]], columns=list("XYZ")).astype(
        np.float32
    )

    X_expected = pd.DataFrame(
        {
            "X": {0: 200.0, 1: 210.0},
            "Y": {0: 140.0, 1: 160.0},
            "Z": {0: 100.0, 1: 125.0},
            "XY_x_45deg": {0: 42.42640687119287, 1: 35.35533905932739},
            "XY_y_45deg": {0: 240.41630560342614, 1: 261.62950903902254},
            "XY_x_60deg": {0: -21.243556529821376, 1: -33.56406460551014},
            "XY_y_60deg": {0: 243.20508075688775, 1: 261.8653347947321},
            "XZ_x_45deg": {0: 70.71067811865477, 1: 60.104076400856556},
            "XZ_y_45deg": {0: 212.13203435596424, 1: 236.8807716974934},
            "XZ_x_60deg": {0: 13.397459621556166, 1: -3.253175473054796},
            "XZ_y_60deg": {0: 223.20508075688775, 1: 244.36533479473212},
        }
    ).astype(np.float32)
    obj = PlaneRotation(
        columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60], dtype=np.float32
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        [[200.0, 140.0, 100.0], [210.0, 160.0, 125.0]], columns=list("XYZ")
    )
    X_expected = pd.DataFrame(
        {
            "X": {0: 200.0, 1: 210.0},
            "Y": {0: 140.0, 1: 160.0},
            "Z": {0: 100.0, 1: 125.0},
            "XY_x_45deg": {0: 42.42640687119287, 1: 35.35533905932739},
            "XY_y_45deg": {0: 240.41630560342614, 1: 261.62950903902254},
            "XY_x_60deg": {0: -21.243556529821376, 1: -33.56406460551014},
            "XY_y_60deg": {0: 243.20508075688775, 1: 261.8653347947321},
            "XZ_x_45deg": {0: 70.71067811865477, 1: 60.104076400856556},
            "XZ_y_45deg": {0: 212.13203435596424, 1: 236.8807716974934},
            "XZ_x_60deg": {0: 13.397459621556166, 1: -3.253175473054796},
            "XZ_y_60deg": {0: 223.20508075688775, 1: 244.36533479473212},
        }
    )
    obj = PlaneRotation(columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60]).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_float32_ks():
    X = ks.DataFrame([[200, 140, 100], [210, 160, 125]], columns=list("XYZ")).astype(
        np.float32
    )
    X_expected = pd.DataFrame(
        {
            "X": {0: 200.0, 1: 210.0},
            "Y": {0: 140.0, 1: 160.0},
            "Z": {0: 100.0, 1: 125.0},
            "XY_x_45deg": {0: 42.42640687119287, 1: 35.35533905932739},
            "XY_y_45deg": {0: 240.41630560342614, 1: 261.62950903902254},
            "XY_x_60deg": {0: -21.243556529821376, 1: -33.56406460551014},
            "XY_y_60deg": {0: 243.20508075688775, 1: 261.8653347947321},
            "XZ_x_45deg": {0: 70.71067811865477, 1: 60.104076400856556},
            "XZ_y_45deg": {0: 212.13203435596424, 1: 236.8807716974934},
            "XZ_x_60deg": {0: 13.397459621556166, 1: -3.253175473054796},
            "XZ_y_60deg": {0: 223.20508075688775, 1: 244.36533479473212},
        }
    ).astype(np.float32)
    obj = PlaneRotation(
        columns=[["X", "Y"], ["X", "Z"]], theta_vec=[45, 60], dtype=np.float32
    ).fit(X)
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
    assert np.allclose(X_new, X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected)


def test_float32_pd(data_float32):
    obj, X, X_expected = data_float32
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_float32_ks(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_float32_pd_np(data_float32):
    obj, X, X_expected = data_float32
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected)


@pytest.mark.koalas
def test_float32_ks_np(data_float32_ks):
    obj, X, X_expected = data_float32_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected)


def test_input():
    with pytest.raises(TypeError):
        _ = PlaneRotation(columns=0, theta_vec=[45, 60])
    with pytest.raises(TypeError):
        _ = PlaneRotation(columns=[["X", "Y"]], theta_vec=0)
    with pytest.raises(TypeError):
        _ = PlaneRotation(columns=[["X", "Y"], ["X", "Z"]], theta_vec=0)
    with pytest.raises(TypeError):
        _ = PlaneRotation(columns=["X", "Y", "X", "Z"], theta_vec=[45, 60])
    with pytest.raises(ValueError):
        _ = PlaneRotation(columns=[], theta_vec=[0])
    with pytest.raises(TypeError):
        _ = PlaneRotation(columns=[["X", "Y"]], theta_vec=[45, "60"])
