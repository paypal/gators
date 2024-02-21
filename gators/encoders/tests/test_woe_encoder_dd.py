# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.woe_encoder import WOEEncoder


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "Q", "W", "W", "W"],
                "B": ["Q", "Q", "W", "W", "W", "W"],
                "C": ["Q", "Q", "Q", "Q", "W", "W"],
                "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
        npartitions=2,
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=2)
    X_expected = pd.DataFrame(
        {
            "A": {
                0: -1.4351,
                1: -1.4351,
                2: -1.4351,
                3: 1.0217,
                4: 1.0217,
                5: 1.0217,
            },
            "B": {
                0: -1.0986,
                1: -1.0986,
                2: 0.5108,
                3: 0.5108,
                4: 0.5108,
                5: 0.5108,
            },
            "C": {
                0: -0.3365,
                1: -0.3365,
                2: -0.3365,
                3: -0.3365,
                4: 0.5108,
                5: 0.5108,
            },
            "D": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0},
        }
    )
    obj = WOEEncoder(regularization=0.5).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_not_inplace():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": ["Q", "Q", "Q", "W", "W", "W"],
                "B": ["Q", "Q", "W", "W", "W", "W"],
                "C": ["Q", "Q", "Q", "Q", "W", "W"],
                "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
        npartitions=2,
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=2)
    X_expected = pd.DataFrame(
        {
            "A__woe": {
                0: -1.4351,
                1: -1.4351,
                2: -1.4351,
                3: 1.0217,
                4: 1.0217,
                5: 1.0217,
            },
            "B__woe": {
                0: -1.0986,
                1: -1.0986,
                2: 0.5108,
                3: 0.5108,
                4: 0.5108,
                5: 0.5108,
            },
            "C__woe": {
                0: -0.3365,
                1: -0.3365,
                2: -0.3365,
                3: -0.3365,
                4: 0.5108,
                5: 0.5108,
            },
        }
    )
    X_expected = pd.concat([X.compute(), X_expected], axis=1)
    obj = WOEEncoder(regularization=0.5, inplace=False).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = dd.from_pandas(
        pd.DataFrame(np.zeros((6, 3)), columns=list("ABC")), npartitions=2
    )
    y = dd.from_pandas(pd.Series([0, 0, 0, 1, 1, 0], name="TARGET"), npartitions=1)
    obj = WOEEncoder().fit(X, y)
    return obj, X, X.compute().copy()


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute().astype(float)
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_dd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X).compute().astype(float)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_dd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy = X.compute().to_numpy()
    X_numpy_new = obj.transform_numpy(X_numpy)
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_data_not_inplace_dd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X).compute()
    X_new[["A", "B", "C"]] = X_new[["A", "B", "C"]].astype("string[pyarrow]")
    X_new[["A__woe", "B__woe", "C__woe"]] = X_new[
        ["A__woe", "B__woe", "C__woe"]
    ].astype(float)
    assert_frame_equal(X_new, X_expected)


def test_data_not_inplace_dd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
