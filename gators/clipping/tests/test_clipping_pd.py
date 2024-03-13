# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.clipping.clipping import Clipping


@pytest.fixture
def data():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
    clip_dict = {"A": [-0.5, 0.5], "B": [-0.5, 0.5], "C": [-100.0, 1.0]}
    obj = Clipping(clip_dict=clip_dict).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.5, 0.4105985019, 0.5],
            "B": [
                0.400157208,
                0.5,
                -0.1513572082976979,
                0.144043571160878,
                0.12167501649282841,
            ],
            "C": [
                0.9787379841057392,
                -0.977277879876411,
                -0.10321885179355784,
                1.0,
                0.4438632327,
            ],
        }
    )
    return obj, X, X_expected


@pytest.fixture
def data_not_inplace():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
    clip_dict = {"A": [-0.5, 0.5], "B": [-0.5, 0.5], "C": [-100.0, 1.0]}
    obj = Clipping(clip_dict=clip_dict, inplace=False).fit(X)
    X_expected = pd.DataFrame(
        {
            "A__clip": [0.5, 0.5, 0.5, 0.4105985019, 0.5],
            "B__clip": [
                0.400157208,
                0.5,
                -0.1513572082976979,
                0.144043571160878,
                0.12167501649282841,
            ],
            "C__clip": [
                0.9787379841057392,
                -0.977277879876411,
                -0.10321885179355784,
                1.0,
                0.4438632327,
            ],
        }
    )
    return obj, X, pd.concat([X, X_expected], axis=1)


@pytest.fixture
def data_partial():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
    clip_dict = {"A": [-0.5, 0.5], "B": [-0.5, 0.5]}
    obj = Clipping(clip_dict=clip_dict).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.5, 0.4105985019, 0.5],
            "B": [
                0.400157208,
                0.5,
                -0.1513572082976979,
                0.144043571160878,
                0.12167501649282841,
            ],
            "C": [
                0.9787379841057392,
                -0.977277879876411,
                -0.10321885179355784,
                1.454274,
                0.4438632327,
            ],
        }
    )
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


def test_not_inplace_pd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_not_inplace_pd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


def test_partial_pd(data_partial):
    obj, X, X_expected = data_partial
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_partial_pd_np(data_partial):
    obj, X, X_expected = data_partial
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


def test_init():
    with pytest.raises(TypeError):
        _ = Clipping(clip_dict=0)
    with pytest.raises(ValueError):
        _ = Clipping(clip_dict={})
    with pytest.raises(TypeError):
        _ = Clipping(clip_dict={"A": [0, 1]}, inplace="yes")
