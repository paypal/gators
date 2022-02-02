# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pyspark.ml.classification import RandomForestClassifier as RFCSpark
from sklearn.datasets import load_iris

from gators.feature_selection.select_from_model import SelectFromModel
from gators.model_building.model import Model
from gators.pipeline.pipeline import Pipeline
from gators.transformers.transformer import Transformer

ks.set_option("compute.default_index_type", "distributed-sequence")

from sklearn.datasets import load_iris

data = load_iris()


class MultiplyTransformer(Transformer):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Transformer.check_dataframe(X)
        return self.multiplier * X

    def transform_numpy(self, X):
        Transformer.check_array(X)
        return self.multiplier * X


class NameTransformer(Transformer):
    def fit(self, X, y=None):
        self.column_names = [f"{c}_new" for c in X.columns]
        return self

    def transform(self, X):
        Transformer.check_dataframe(X)
        return X.rename(columns=dict(zip(X.columns, self.column_names)))

    def transform_numpy(self, X):
        Transformer.check_array(X)
        return X


@pytest.fixture
def pipeline_example():
    X = ks.DataFrame(data["data"], columns=data["feature_names"])
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
    ]
    pipe = Pipeline(steps)
    return pipe, X


@pytest.fixture
def pipeline_with_model_example():
    X = ks.DataFrame(data["data"], columns=data["feature_names"])
    y = ks.Series(data["target"], name="TARGET")

    model = RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
        ["Estimator", Model(model)],
    ]
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X


@pytest.mark.koalas
def test_pipeline_fit_and_transform_ks(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    X_new = pipe.transform(X)
    assert X_new.shape == (150, 4)
    assert list(X_new.columns) == [
        "sepal length (cm)_new",
        "sepal width (cm)_new",
        "petal length (cm)_new",
        "petal width (cm)_new",
    ]


@pytest.mark.koalas
def test_fit_transform_pipeline_ks(pipeline_example):
    pipe, X = pipeline_example
    X_new = pipe.fit_transform(X)
    assert X_new.shape == (150, 4)
    assert list(X_new.columns) == [
        "sepal length (cm)_new",
        "sepal width (cm)_new",
        "petal length (cm)_new",
        "petal width (cm)_new",
    ]


@pytest.mark.koalas
def test_pipeline_predict_ks(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict(X)
    assert y_pred.shape == (150,)


@pytest.mark.koalas
def test_pipeline_predict_proba_ks(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict_proba(X)
    assert y_pred.shape == (150,)


@pytest.mark.koalas
def test_pipeline_np_ks(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    X_numpy_new = pipe.transform_numpy(X.to_numpy())
    assert X_numpy_new.shape == (150, 4)
