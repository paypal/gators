# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.feature_generation.cluster_statistics import ClusterStatistics

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame({"A": [0.0, 3.0, 6.0], "B": [1.0, 4.0, 7.0], "C": [2.0, 5.0, 8.0]})
    clusters_dict = {
        "cluster_name_a": list("AB"),
        "cluster_name_b": list("AC"),
        "cluster_name_c": list("BC"),
    }
    obj = ClusterStatistics(clusters_dict=clusters_dict).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "cluster_name_a__mean": [0.5, 3.5, 6.5],
            "cluster_name_a__std": [
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
            ],
            "cluster_name_b__mean": [1.0, 4.0, 7.0],
            "cluster_name_b__std": [
                1.4142135623730951,
                1.4142135623730951,
                1.4142135623730951,
            ],
            "cluster_name_c__mean": [1.5, 4.5, 7.5],
            "cluster_name_c__std": [
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
            ],
        }
    )
    return obj, X, X_expected


@pytest.fixture
def data_object_ks():
    X = ks.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "D": ["a", "b", "c"],
        }
    )
    clusters_dict = {
        "cluster_name_a": list("AB"),
        "cluster_name_b": list("AC"),
        "cluster_name_c": list("BC"),
    }
    obj = ClusterStatistics(clusters_dict=clusters_dict).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "D": ["a", "b", "c"],
            "cluster_name_a__mean": [0.5, 3.5, 6.5],
            "cluster_name_a__std": [
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
            ],
            "cluster_name_b__mean": [1.0, 4.0, 7.0],
            "cluster_name_b__std": [
                1.4142135623730951,
                1.4142135623730951,
                1.4142135623730951,
            ],
            "cluster_name_c__mean": [1.5, 4.5, 7.5],
            "cluster_name_c__std": [
                0.7071067811865476,
                0.7071067811865476,
                0.7071067811865476,
            ],
        }
    )

    return obj, X, X_expected


@pytest.fixture
def data_names_ks():
    X = ks.DataFrame({"A": [0.0, 3.0, 6.0], "B": [1.0, 4.0, 7.0], "C": [2.0, 5.0, 8.0]})
    clusters_dict = {
        "cluster_name_a": list("AB"),
        "cluster_name_b": list("AC"),
        "cluster_name_c": list("BC"),
    }
    obj = ClusterStatistics(
        clusters_dict=clusters_dict,
        column_names=["a_mean", "a_std", "bb_mean", "bb_std", "ccc_mean", "ccc_std"],
    ).fit(X)
    X_expected = pd.DataFrame(
        {
            "A": [0.0, 3.0, 6.0],
            "B": [1.0, 4.0, 7.0],
            "C": [2.0, 5.0, 8.0],
            "a_mean": [0.5, 3.5, 6.5],
            "a_std": [0.7071067811865476, 0.7071067811865476, 0.7071067811865476],
            "bb_mean": [1.0, 4.0, 7.0],
            "bb_std": [1.4142135623730951, 1.4142135623730951, 1.4142135623730951],
            "ccc_mean": [1.5, 4.5, 7.5],
            "ccc_std": [0.7071067811865476, 0.7071067811865476, 0.7071067811865476],
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
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_object_ks(data_object_ks):
    obj, X, X_expected = data_object_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_object_ks_np(data_object_ks):
    obj, X, X_expected = data_object_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values)
    assert_frame_equal(X_new, X_expected)


@pytest.mark.koalas
def test_names_ks(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_names_np(data_names_ks):
    obj, X, X_expected = data_names_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    X_expected = pd.DataFrame(X_expected.values.astype(np.float64))
    assert_frame_equal(X_new, X_expected)
