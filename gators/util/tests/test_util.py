# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.util import util

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    return pd.DataFrame(
        {"A": 1.1, "B": 2.1, "C": 3.1, "D": 4, "E": 5, "F": "a"}, index=[0]
    )


@pytest.fixture
def data_ks():
    return ks.DataFrame(
        {"A": 1.1, "B": 2.1, "C": 3.1, "D": 4, "E": 5, "F": "a"}, index=[0]
    )


def test_get_bounds():
    assert np.allclose(util.get_bounds(np.float32), [-3.4028235e38, 3.4028235e38])
    assert np.allclose(util.get_bounds(np.int32), [-2147483648, 2147483647])
    assert np.allclose(util.get_bounds(np.int16), [-32768, 32767])


def test_get_datatype_columns_float_pd(data):
    X = data
    float_columns = util.get_datatype_columns(X, np.float64)
    assert float_columns == ["A", "B", "C"]


def test_get_datatype_columns_int_pd(data):
    X = data
    int_columns = util.get_datatype_columns(X, np.int64)
    assert int_columns == ["D", "E"]


def test_get_datatype_columns_object_pd(data):
    X = data
    object_columns = util.get_datatype_columns(X, object)
    assert object_columns == ["F"]


def test_get_float_only_columns_pd(data):
    X = data
    float_only_columns = util.get_float_only_columns(X)
    assert float_only_columns == ["A", "B", "C"]


def test_get_float_only_columns_pd_no(data):
    X = pd.DataFrame({"F": "a"}, index=[0])
    float_only_columns = util.get_float_only_columns(X)
    assert float_only_columns == []


def test_get_int_only_columns_pd(data):
    X = data
    int_only_columns = util.get_int_only_columns(X)
    assert int_only_columns == ["D", "E"]


def test_get_int_only_columns_pd_no(data):
    X = pd.DataFrame({"A": "a", "B": "b", "C": "c", "F": "f"}, index=[0])
    int_only_columns = util.get_int_only_columns(X)
    assert int_only_columns == []


def test_get_idx_columns():
    columns = ["A", "B", "C", "D", "E"]
    selected_columns = ["A", "C", "E"]
    indx = util.get_idx_columns(columns, selected_columns)
    assert np.allclose(indx, np.array([0, 2, 4]))


def test_exclude_idx_columns():
    columns = ["A", "B", "C", "D", "E"]
    selected_columns = ["A", "C", "E"]
    idx = util.exclude_idx_columns(columns, selected_columns)
    assert np.allclose(idx, np.array([1, 3]))


def test_concat_pd(data):
    X = data
    assert_frame_equal(
        util.concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1),
        pd.concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1),
    )


@pytest.mark.koalas
def test_get_datatype_columns_float_ks(data_ks):
    X = data_ks
    float_columns = util.get_datatype_columns(X, np.float64)
    assert float_columns == ["A", "B", "C"]


@pytest.mark.koalas
def test_get_datatype_columns_int_ks(data_ks):
    X = data_ks
    int_columns = util.get_datatype_columns(X, np.int64)
    assert int_columns == ["D", "E"]


@pytest.mark.koalas
def test_get_datatype_columns_object_ks(data_ks):
    X = data_ks
    object_columns = util.get_datatype_columns(X, object)
    assert object_columns == ["F"]


@pytest.mark.koalas
def test_get_float_only_columns_ks(data_ks):
    X = data_ks
    float_only_columns = util.get_float_only_columns(X)
    assert float_only_columns == ["A", "B", "C"]


@pytest.mark.koalas
def test_get_int_only_columns_ks(data_ks):
    X = data_ks
    int_only_columns = util.get_int_only_columns(X)
    assert int_only_columns == ["D", "E"]


@pytest.mark.koalas
def test_concat_ks(data_ks):
    X = data_ks
    assert_frame_equal(
        util.concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1).to_pandas(),
        ks.concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1).to_pandas(),
    )


@pytest.mark.koalas
def test_generate_spark_dataframe():
    X = ks.DataFrame({"A": [1, 2, 3], "B": [100, 50, 50]})
    y = ks.Series([0, 1, 0], name="TARGET")
    ks_expected = ks.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [100, 50, 50],
            "TARGET": [0, 1, 0],
            "features": [[1.0, 100.0], [2.0, 50.0], [3.0, 50.0]],
        }
    )
    spark_df = util.generate_spark_dataframe(X=X, y=y)
    spark_df_as_ks = ks.DataFrame(spark_df)
    assert_frame_equal(ks_expected.to_pandas(), spark_df_as_ks.to_pandas())


@pytest.mark.koalas
def test_generate_spark_dataframe_no_target():
    X = ks.DataFrame({"A": [1, 2, 3], "B": [100, 50, 50]})
    ks_expected = ks.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [100, 50, 50],
            "features": [[1.0, 100.0], [2.0, 50.0], [3.0, 50.0]],
        }
    )
    spark_df = util.generate_spark_dataframe(X=X)
    spark_df_as_ks = ks.DataFrame(spark_df)
    assert_frame_equal(ks_expected.to_pandas(), spark_df_as_ks.to_pandas())


def test_flatten_list():
    assert util.flatten_list([1, 2, [3, 4]]) == [1, 2, 3, 4]
