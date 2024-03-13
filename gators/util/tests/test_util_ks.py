# License: Apache-2.0
import pyspark.pandas as ps
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.util import util

ps.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    return ps.DataFrame(
        {"A": 1.1, "B": 2.1, "C": 3.1, "D": 4, "E": 5, "F": "a"}, index=[0]
    )


def test_concat_ks(data_ks):
    X = data_ks
    assert_frame_equal(
        util.get_function(X).concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1),
        pd.concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1),
    )


@pytest.mark.pyspark
def test_get_datatype_columns_object_ks(data_ks):
    X = data_ks
    object_columns = util.get_datatype_columns(X, object)
    assert object_columns == ["F"]


@pytest.mark.pyspark
def test_concat_ks(data_ks):
    X = data_ks
    assert_frame_equal(
        util.get_function(X)
        .concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1)
        .to_pandas(),
        ps.concat([X, X[["A"]].rename(columns={"A": "AA"})], axis=1).to_pandas(),
    )


@pytest.mark.pyspark
def test_fillna(data_ks):
    X = data_ks
    _ = util.get_function(X).fillna(X.astype(int), value={"A": 0})
    assert True
