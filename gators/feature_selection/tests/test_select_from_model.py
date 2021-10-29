# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.ml.classification import RandomForestClassifier as RFCSpark
from xgboost import XGBClassifier

from gators.feature_selection.select_from_model import SelectFromModel

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
            "B": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name="TARGET")
    X_expected = X[["A", "B"]].copy()
    model = XGBClassifier(
        random_state=0,
        subsample=1.0,
        n_estimators=2,
        max_depth=2,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    obj = SelectFromModel(model=model, k=2).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name="TARGET")
    X_expected = X[["A"]].to_pandas().copy()
    model = RFCSpark(numTrees=2, maxDepth=1, labelCol=y.name, seed=0)
    obj = SelectFromModel(model=model, k=2).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_combined():
    X = ks.DataFrame(
        {
            "A": [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name="TARGET")
    X_expected = X[["A"]].to_pandas().copy()
    model = XGBClassifier(
        random_state=0,
        subsample=1.0,
        n_estimators=2,
        max_depth=2,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    obj = SelectFromModel(model=model, k=2).fit(X, y)
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
    assert X_expected.shape == X_new.shape


@pytest.mark.koalas
def test_ks_pd(data_combined):
    obj, X, X_expected = data_combined
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert_frame_equal(X_new, pd.DataFrame(X_expected.to_numpy()))


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert X_expected.shape == X_new.shape


def test_init():
    with pytest.raises(TypeError):
        _ = SelectFromModel(model=XGBClassifier(), k="a")

    class Model:
        pass

    with pytest.raises(TypeError):
        _ = SelectFromModel(model=Model(), k=2)
