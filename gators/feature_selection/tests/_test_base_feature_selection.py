# License: Apache-2.0
import os

import databricks.koalas as ks
import pandas as pd
import pytest
from pyspark.ml.classification import GBTClassifier as GBTCSpark
from pyspark.ml.classification import RandomForestClassifier as RFCSpark
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier

from gators.feature_selection.information_value import InformationValue
from gators.feature_selection.select_from_model import SelectFromModel
from gators.feature_selection.select_from_models import SelectFromModels
from gators.feature_selection.variance_filter import VarianceFilter


@pytest.fixture
def data():
    dataset = load_breast_cancer(return_X_y=False)
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    return X, _, y


@pytest.mark.koalas
@pytest.fixture
def data():
    dataset = load_breast_cancer(return_X_y=False)
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    return X, ks.from_pandas(X), y


def test_variance_filter(data):
    X, _, y = data
    max_variance = 1
    variance_filter = VarianceFilter(max_variance=max_variance)
    _ = variance_filter.fit_transform(X, y).columns
    saved_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "perimeter error",
        "area error",
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
    ]
    assert sorted(variance_filter.selected_columns) == sorted(saved_features)


@pytest.mark.koalas
def test_variance_filter_koalas(data):
    _, X, y = data
    max_variance = 1
    variance_filter = VarianceFilter(max_variance=max_variance)
    _ = variance_filter.fit_transform(X, y).columns
    saved_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "perimeter error",
        "area error",
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
    ]
    assert sorted(variance_filter.selected_columns) == sorted(saved_features)


def test_information_value(data):
    X, _, y = data
    iv_selector = InformationValue(k=10)
    _ = iv_selector.fit(X, y)
    saved_features = [
        "worst perimeter",
        "worst area",
        "worst radius",
        "mean concave points",
        "worst concave points",
        "mean concavity",
        "mean perimeter",
        "worst concavity",
        "mean area",
        "mean radius",
    ]
    assert sorted(iv_selector.selected_columns) == sorted(saved_features)


@pytest.mark.koalas
def test_information_value_koalas(data):
    _, X, y = data
    iv_selector = InformationValue(k=10)
    _ = iv_selector.fit(X, y)
    saved_features = [
        "worst perimeter",
        "worst area",
        "worst radius",
        "mean concave points",
        "worst concave points",
        "mean concavity",
        "mean perimeter",
        "worst concavity",
        "mean area",
        "mean radius",
    ]
    assert sorted(iv_selector.selected_columns) == sorted(saved_features)


def test_select_from_model_with_xgb(data):
    X, _, y = data
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=0)
    xgb_selector = SelectFromModel(model=model, k=10)
    X_new = xgb_selector.fit_transform(X, y)
    saved_features = [
        "worst radius",
        "worst perimeter",
        "mean concave points",
        "worst concave points",
        "concave points error",
        "worst concavity",
        "mean texture",
        "worst texture",
        "worst area",
        "smoothness error",
    ]
    assert sorted(xgb_selector.selected_columns) == sorted(saved_features)


@pytest.mark.koalas
def test_select_from_model_random_forest_koalas(data):
    X, X_ks, y = data
    y_ks = ks.from_pandas(pd.Series(y))
    model = RFCSpark(numTrees=50, maxDepth=5, seed=0)
    rf_selector = SelectFromModel(model=model, k=10)
    _ = rf_selector.fit_transform(X_ks, y_ks)
    selected_columns = [
        "worst perimeter",
        "worst concave points",
        "worst radius",
        "mean concave points",
        "worst area",
        "area error",
        "mean radius",
        "mean perimeter",
        "mean area",
        "mean concavity",
    ]
    assert rf_selector.selected_columns == selected_columns


@pytest.mark.koalas
def test_select_from_models_koalas(data):
    X, X_ks, y = data
    y_ks = ks.from_pandas(pd.Series(y))
    gbt = GBTCSpark(maxIter=5, maxDepth=5, seed=0)
    models_selector = SelectFromModels([gbt], k=10)
    X_new = models_selector.fit_transform(X=X_ks, y=y_ks)
    saved_features = [
        "worst perimeter",
        "mean concave points",
        "worst texture",
        "worst concave points",
        "mean texture",
        "texture error",
        "worst concavity",
        "mean symmetry",
        "mean smoothness",
        "concave points error",
    ]
    assert sorted(models_selector.selected_columns) == sorted(saved_features)
