# License: Apache-2.0
import pyspark.pandas as ps
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.woe_encoder import WOEEncoder

ps.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ps.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = ps.Series([0, 0, 0, 1, 1, 0], name="TARGET")
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
def data_no_cat_ks():
    X = ps.DataFrame(
        np.zeros((6, 3)),
        columns=list("ABC"),
    )
    y = ps.Series([0, 0, 0, 1, 1, 0], name="TARGET")
    obj = WOEEncoder().fit(X, y)
    return obj, X, X.to_pandas().copy()


@pytest.mark.pyspark
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.pyspark
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.pyspark
def test_no_cat_ks(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.pyspark
def test_no_cat_ks_np(data_no_cat_ks):
    obj, X, X_expected = data_no_cat_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


@pytest.fixture
def data_not_inplace():
    X = ps.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = ps.Series([0, 0, 0, 1, 1, 0], name="TARGET")
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
    X_expected = pd.concat([X.to_pandas(), X_expected], axis=1)
    obj = WOEEncoder(regularization=0.5, inplace=False).fit(X, y)
    return obj, X, X_expected


@pytest.mark.pyspark
def test_data_not_inplace_pd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.pyspark
def test_data_not_inplace_pd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
