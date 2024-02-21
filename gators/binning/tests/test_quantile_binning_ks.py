# License: Apache-2.0
import pyspark.pandas as ps
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import QuantileBinning

ps.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    n_bins = 4
    X = ps.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": [
                "(-inf, 7.92)",
                "[53.1, inf)",
                "[7.92, 8.05)",
                "[53.1, inf)",
                "[8.05, 53.1)",
                "[8.05, 53.1)",
            ],
            "B__bin": [
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
            ],
            "D__bin": [
                "(-inf, 26.0)",
                "[30.0, inf)",
                "[26.0, 27.2)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[27.2, 30.0)",
            ],
            "F__bin": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace_ks():
    n_bins = 4
    X = ps.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [
                "(-inf, 7.92)",
                "[53.1, inf)",
                "[7.92, 8.05)",
                "[53.1, inf)",
                "[8.05, 53.1)",
                "[8.05, 53.1)",
            ],
            "B": [
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
            ],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [
                "(-inf, 26.0)",
                "[30.0, inf)",
                "[26.0, 27.2)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[27.2, 30.0)",
            ],
            "F": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_num_ks():
    n_bins = 3
    X = ps.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_ks():
    n_bins = 4
    X = ps.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
            "A__bin": [
                "(-inf, 7.92)",
                "[53.1, inf)",
                "[7.92, 8.05)",
                "[53.1, inf)",
                "[8.05, 53.1)",
                "[8.05, 53.1)",
            ],
            "B__bin": [
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
            ],
            "D__bin": [
                "(-inf, 26.0)",
                "[30.0, inf)",
                "[26.0, 27.2)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[27.2, 30.0)",
            ],
            "F__bin": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_inplace_ks():
    n_bins = 4
    X = ps.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 30.0, 30.0, 27.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": [
                "(-inf, 7.92)",
                "[53.1, inf)",
                "[7.92, 8.05)",
                "[53.1, inf)",
                "[8.05, 53.1)",
                "[8.05, 53.1)",
            ],
            "B": [
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
                "(-inf, inf)",
            ],
            "D": [
                "(-inf, 26.0)",
                "[30.0, inf)",
                "[26.0, 27.2)",
                "[30.0, inf)",
                "[30.0, inf)",
                "[27.2, 30.0)",
            ],
            "F": [
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "(-inf, 2.0)",
                "[2.0, inf)",
                "[2.0, inf)",
            ],
        }
    )
    obj = QuantileBinning(n_bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.mark.pyspark
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.pyspark
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.pyspark
def test_no_num_ks(data_no_num_ks):
    obj, X, X_expected = data_no_num_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


@pytest.mark.pyspark
def test_no_num_ks_np(data_no_num_ks):
    obj, X, X_expected = data_no_num_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.pyspark
def test_num_ks(data_num_ks):
    obj, X, X_expected = data_num_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


@pytest.mark.pyspark
def test_num_ks_np(data_num_ks):
    obj, X, X_expected = data_num_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.pyspark
def test_inplace_ks(data_inplace_ks):
    obj, X, X_expected = data_inplace_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


@pytest.mark.pyspark
def test_inplace_ks_np(data_inplace_ks):
    obj, X, X_expected = data_inplace_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.pyspark
def test_inplace_num_ks(data_num_inplace_ks):
    obj, X, X_expected = data_num_inplace_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


@pytest.mark.pyspark
def test_inplace_num_ks_np(data_num_inplace_ks):
    obj, X, X_expected = data_num_inplace_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))
