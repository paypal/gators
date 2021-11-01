# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.is_equal import IsEqual

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
    )

    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16():
    X = pd.DataFrame(
        {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
    ).astype(np.int16)

    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    ).astype(np.int16)
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_obj():
    X = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["a", "f", "e"],
            "C": ["a", "p", "d"],
            "D": [1, 2, 3],
        }
    )

    X_expected = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["a", "f", "e"],
            "C": ["a", "p", "d"],
            "D": [1, 2, 3],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names():
    X = pd.DataFrame(
        {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
    )

    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A==B": [1.0, 0.0, 0.0],
            "A==C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(
        columns_a=list("AA"), columns_b=list("BC"), column_names=["A==B", "A==C"]
    ).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
    )
    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_int16_ks():
    X = ks.DataFrame(
        {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
    ).astype(np.int16)
    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    ).astype(np.int16)
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_obj_ks():
    X = ks.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["a", "f", "e"],
            "C": ["a", "p", "d"],
            "D": [1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["a", "f", "e"],
            "C": ["a", "p", "d"],
            "D": [1, 2, 3],
            "A__is__B": [1.0, 0.0, 0.0],
            "A__is__C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(columns_a=list("AA"), columns_b=list("BC")).fit(X)
    return obj, X, X_expected


@pytest.fixture
def data_names_ks():
    X = ks.DataFrame(
        {"A": [99.0, 1.0, 2.0], "B": [99.0, 4.0, 5.0], "C": [99.0, 7.0, 8.0]}
    )

    X_expected = pd.DataFrame(
        {
            "A": [99.0, 1.0, 2.0],
            "B": [99.0, 4.0, 5.0],
            "C": [99.0, 7.0, 8.0],
            "A==B": [1.0, 0.0, 0.0],
            "A==C": [1.0, 0.0, 0.0],
        }
    )
    obj = IsEqual(
        columns_a=list("AA"), columns_b=list("BC"), column_names=["A==B", "A==C"]
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
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)


def test_int16_pd(data_int16):
    obj, X, X_expected = data_int16
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_int16_ks(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_int16_pd_np(data_int16):
    obj, X, X_expected = data_int16
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_int16_ks_np(data_int16_ks):
    obj, X, X_expected = data_int16_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_obj(data_obj):
    obj, X, X_expected = data_obj
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_obj_ks(data_obj_ks):
    obj, X, X_expected = data_obj_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_obj_np(data_obj):
    obj, X, X_expected = data_obj
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_obj_ks_np(data_obj_ks):
    obj, X, X_expected = data_obj_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


def test_names_pd(data_names):
    obj, X, X_expected = data_names
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


def test_names_pd_np(data_names):
    obj, X, X_expected = data_names
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks_np(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)


def test_input():
    with pytest.raises(TypeError):
        _ = IsEqual(columns_a=0, columns_b=["B"])
    with pytest.raises(TypeError):
        _ = IsEqual(columns_a=["A"], columns_b=0)
    with pytest.raises(TypeError):
        _ = IsEqual(columns_a=["A"], columns_b=["B"], column_names=0)
    with pytest.raises(ValueError):
        _ = IsEqual(columns_a=["A"], columns_b=["B", "C"])
    with pytest.raises(ValueError):
        _ = IsEqual(columns_a=["A"], columns_b=["B"], column_names=["x", "y"])
    with pytest.raises(ValueError):
        _ = IsEqual(columns_a=[], columns_b=[])
