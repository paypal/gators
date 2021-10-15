# License: Apache-2.0
from gators.feature_selection.select_from_models import SelectFromModels
from pyspark.ml.classification import RandomForestClassifier as RFCSpark
from xgboost import XGBClassifier, XGBRFClassifier
from pandas.testing import assert_frame_equal
import pytest
import numpy as np
import pandas as pd
import databricks.koalas as ks
ks.set_option('compute.default_index_type', 'distributed-sequence')


@pytest.fixture
def data():
    X = pd.DataFrame({
        'A':
            {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0,
                5: 28.11, 6: 54.0, 7: 2.0, 8: 27.0, 9: 14.0},
        'B':
            {0: 7.25, 1: 71.28, 2: 7.92, 3: 53.1, 4: 8.05,
                5: 8.46, 6: 51.86, 7: 21.08, 8: 11.13, 9: 30.07},
        'C':
            {0: 3.0, 1: 1.0, 2: 3.0, 3: 1.0, 4: 3.0,
                5: 3.0, 6: 1.0, 7: 3.0, 8: 3.0, 9: 2.0}})
    y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    X_expected = X[['A', 'B']].copy()
    model1 = XGBClassifier(
        random_state=0,
        subsample=1.,
        n_estimators=2,
        max_depth=2,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model2 = XGBRFClassifier(
        random_state=0,
        subsample=1.,
        n_estimators=2,
        max_depth=2,
        eval_metric='logloss',
        use_label_encoder=False
    )
    obj = SelectFromModels(models=[model1, model2], k=2).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_ks():
    X = ks.DataFrame({
        'A':
            {0: 22.0, 1: 38.0, 2: 26.0, 3: 35.0, 4: 35.0,
                5: 28.11, 6: 54.0, 7: 2.0, 8: 27.0, 9: 14.0},
        'B':
            {0: 7.25, 1: 71.28, 2: 7.92, 3: 53.1, 4: 8.05,
                5: 8.46, 6: 51.86, 7: 21.08, 8: 11.13, 9: 30.07},
        'C':
            {0: 3.0, 1: 1.0, 2: 3.0, 3: 1.0, 4: 3.0,
                5: 3.0, 6: 1.0, 7: 3.0, 8: 3.0, 9: 2.0}})
    y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    X_expected = X[['A', 'B']].to_pandas().copy()
    model1_ks = RFCSpark(
        numTrees=2, maxDepth=2, labelCol=y.name, seed=0)
    mode2_ks = RFCSpark(
        numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    obj = SelectFromModels(
        models=[model1_ks, mode2_ks], k=2).fit(X, y)
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
    X_new = pd.DataFrame(X_numpy_new)
    assert_frame_equal(X_new, pd.DataFrame(X_expected.to_numpy()))


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(np.float64))


def test_init():
    with pytest.raises(TypeError):
        _ = SelectFromModels(models=0, k='a')

    with pytest.raises(TypeError):
        _ = SelectFromModels(models=[XGBClassifier()], k='a')

    class Model():
        pass
    with pytest.raises(TypeError):
        _ = SelectFromModels(models=[Model()], k=2)
