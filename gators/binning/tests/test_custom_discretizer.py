# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.binning import CustomDiscretizer

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: 7.25, 1: 71.2833, 2: 7.925, 3: 53.1, 4: 8.05, 5: 8.4583},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "C": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"},
            "D": {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0, 5: 31.2},
            "F": {0: 3, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
            "A__bin": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "D__bin": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F__bin": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X[list("ABDF")] = X[list("ABDF")].astype(np.int16)
    X_expected = pd.DataFrame(
        {
            "A": {0: 7, 1: 71, 2: 7, 3: 53, 4: 8, 5: 8},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "C": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"},
            "D": {0: 22, 1: 38, 2: 26, 3: 35, 4: 35, 5: 31},
            "F": {0: 3, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
            "A__bin": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "0.0", 5: "0.0"},
            "D__bin": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F__bin": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    X_expected[list("ABDF")] = X_expected[list("ABDF")].astype(np.int16)
    bins = {"A": [-1000, 8, 40, 1000], "D": [-1000, 30, 1000], "F": [-1000, 1.0, 1000]}
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_num():
    X = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "C": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"},
            "D": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: 7.25, 1: 71.2833, 2: 7.925, 3: 53.1, 4: 8.05, 5: 8.4583},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "D": {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0, 5: 31.2},
            "F": {0: 3, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
            "A__bin": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "D__bin": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F__bin": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_inplace():
    X = pd.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "D": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: 7.25, 1: 71.2833, 2: 7.925, 3: 53.1, 4: 8.05, 5: 8.4583},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "C": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"},
            "D": {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0, 5: 31.2},
            "F": {0: 3, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
            "A__bin": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "D__bin": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F__bin": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16_ks():
    X = ks.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X[list("ABDF")] = X[list("ABDF")].astype(np.int16)
    X_expected = pd.DataFrame(
        {
            "A": {0: 7, 1: 71, 2: 7, 3: 53, 4: 8, 5: 8},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "C": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"},
            "D": {0: 22, 1: 38, 2: 26, 3: 35, 4: 35, 5: 31},
            "F": {0: 3, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
            "A__bin": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "0.0", 5: "0.0"},
            "D__bin": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F__bin": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    X_expected[list("ABDF")] = X_expected[list("ABDF")].astype(np.int16)
    bins = {"A": [-1000, 8, 40, 1000], "D": [-1000, 30, 1000], "F": [-1000, 1.0, 1000]}
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_no_num_ks():
    X = ks.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    X_expected = pd.DataFrame({"C": ["a", "b", "c", "d", "e", "f"]})
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_inplace_ks():
    X = ks.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "C": ["a", "b", "c", "d", "e", "f"],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "C": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"},
            "D": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins, inplace=True).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_ks():
    X = ks.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: 7.25, 1: 71.2833, 2: 7.925, 3: 53.1, 4: 8.05, 5: 8.4583},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "D": {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0, 5: 31.2},
            "F": {0: 3, 1: 1, 2: 2, 3: 1, 4: 2, 5: 3},
            "A__bin": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "D__bin": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F__bin": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_num_inplace_ks():
    X = ks.DataFrame(
        {
            "A": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583],
            "B": [1, 1, 0, 1, 0, 0],
            "D": [22.0, 38.0, 26.0, 35.0, 35.0, 31.2],
            "F": [3, 1, 2, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": {0: "0.0", 1: "2.0", 2: "0.0", 3: "2.0", 4: "1.0", 5: "1.0"},
            "B": {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0},
            "D": {0: "0.0", 1: "1.0", 2: "0.0", 3: "1.0", 4: "1.0", 5: "1.0"},
            "F": {0: "1.0", 1: "0.0", 2: "1.0", 3: "0.0", 4: "1.0", 5: "1.0"},
        }
    )
    bins = {
        "A": [-np.inf, 8.0, 40.0, np.inf],
        "D": [-np.inf, 30, np.inf],
        "F": [-np.inf, 1.0, np.inf],
    }
    obj = CustomDiscretizer(bins, inplace=True).fit(X)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_int16_pd(data_int16):
    obj, X, X_expected = data_int16
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_int16_ks(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_int16_pd_np(data_int16):
    obj, X, X_expected = data_int16
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_int16_ks_np(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_no_num_pd(data_no_num):
    obj, X, X_expected = data_no_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_no_num_ks(data_no_num_ks):
    obj, X, X_expected = data_no_num_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_no_num_pd_np(data_no_num):
    obj, X, X_expected = data_no_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_no_num_ks_np(data_no_num_ks):
    obj, X, X_expected = data_no_num_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_num_pd(data_num):
    obj, X, X_expected = data_num
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_num_ks(data_num_ks):
    obj, X, X_expected = data_num_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_num_pd_np(data_num):
    obj, X, X_expected = data_num
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_num_ks_np(data_num_ks):
    obj, X, X_expected = data_num_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


# # inplace


def test_inplace_pd(data_inplace):
    obj, X, X_expected = data_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_inplace_ks(data_inplace_ks):
    obj, X, X_expected = data_inplace_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_inplace_pd_np(data_inplace):
    obj, X, X_expected = data_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_inplace_ks_np(data_inplace_ks):
    obj, X, X_expected = data_inplace_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_inplace_num_pd(data_num_inplace):
    obj, X, X_expected = data_num_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_inplace_num_ks(data_num_inplace_ks):
    obj, X, X_expected = data_num_inplace_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_inplace_num_pd_np(data_num_inplace):
    obj, X, X_expected = data_num_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_inplace_num_ks_np(data_num_inplace_ks):
    obj, X, X_expected = data_num_inplace_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


def test_init():
    with pytest.raises(TypeError):
        _ = CustomDiscretizer(bins="a")
    with pytest.raises(TypeError):
        _ = CustomDiscretizer(bins={"A": [-np.inf, np.inf]}, inplace="a")
