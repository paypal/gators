# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.clipping.clipping import Clipping


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": {0: 1.76, 1: 2.24, 2: 0.95, 3: 0.41, 4: 0.76},
                "B": {0: 0.4, 1: 1.87, 2: -0.15, 3: 0.14, 4: 0.12},
                "C": {0: 0.98, 1: -0.98, 2: -0.1, 3: 1.45, 4: 0.44},
            }
        ),
        npartitions=1,
    )
    clip_dict = {"A": [-0.5, 0.5], "B": [-0.5, 0.5], "C": [-100.0, 1.0]}
    obj = Clipping(clip_dict=clip_dict).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.41, 4: 0.5},
            "B": {0: 0.4, 1: 0.5, 2: -0.15, 3: 0.14, 4: 0.12},
            "C": {0: 0.98, 1: -0.98, 2: -0.1, 3: 1.0, 4: 0.44},
        }
    )
    return obj, X, X_expected


@pytest.fixture
def data_partial():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": {0: 1.76, 1: 2.24, 2: 0.95, 3: 0.41, 4: 0.76},
                "B": {0: 0.4, 1: 1.87, 2: -0.15, 3: 0.14, 4: 0.12},
                "C": {0: 0.98, 1: -0.98, 2: -0.1, 3: 1.45, 4: 0.44},
            }
        ),
        npartitions=1,
    )
    clip_dict = {"A": [-0.5, 0.5]}
    obj = Clipping(clip_dict=clip_dict).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.41, 4: 0.5},
            "B": {0: 0.4, 1: 1.87, 2: -0.15, 3: 0.14, 4: 0.12},
            "C": {0: 0.98, 1: -0.98, 2: -0.1, 3: 1.45, 4: 0.44},
        }
    )
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute()
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert np.allclose(X_new, X_expected.to_numpy())


# def test_partial_dd(data_partial):
#     obj, X, X_expected = data_partial
#     X_new = obj.transform(X).compute()
#     assert_frame_equal(X_new, X_expected)


# def test_partial_dd_np(data_partial):
#     obj, X, X_expected = data_partial
#     X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
#     X_new = pd.DataFrame(X_numpy_new)
#     assert np.allclose(X_new, X_expected.to_numpy())
