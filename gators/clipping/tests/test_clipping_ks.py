# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.clipping.clipping import Clipping

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    np.random.seed(0)
    X = ks.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
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
def data_int16_ks():
    np.random.seed(0)
    X = ks.DataFrame(5 * np.random.randn(5, 3), columns=list("ABC")).astype(np.int16)
    clip_dict = {"A": [-5, 2], "B": [-1, 3], "C": [-2, 5]}
    obj = Clipping(clip_dict=clip_dict, dtype=np.int16).fit(X)
    X_expected = pd.DataFrame(
        {"A": [2, 2, 2, 2, 2], "B": [2, 3, 0, 0, 0], "C": [4, -2, 0, 5, 2]}
    ).astype(np.int16)
    return obj, X, X_expected


@pytest.fixture
def data_partial_ks():
    np.random.seed(0)
    X = ks.DataFrame(np.random.randn(5, 3), columns=list("ABC"))
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


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


@pytest.mark.koalas
def test_int16_ks(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_int16_ks_np(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


@pytest.mark.koalas
def test_partial_ks(data_partial_ks):
    obj, X, X_expected = data_partial_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_partial_ks_np(data_partial_ks):
    obj, X, X_expected = data_partial_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())
