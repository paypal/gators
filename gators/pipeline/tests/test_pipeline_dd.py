# License: Apache-2.0
import pytest
import dask.dataframe as dd
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from gators.model_building.model import Model
from gators.pipeline.pipeline import Pipeline
from gators.transformers.transformer import Transformer

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
        return X.rename(columns=dict(zip(X.columns, self.column_names)))

    def transform_numpy(self, X):
        return X


@pytest.fixture
def pipeline_example():
    X = dd.from_pandas(
        pd.DataFrame(data["data"], columns=data["feature_names"]), npartitions=1
    )
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
    ]
    pipe = Pipeline(steps)
    return pipe, X


@pytest.fixture
def pipeline_with_model_example():
    X = dd.from_pandas(
        pd.DataFrame(data["data"], columns=data["feature_names"]), npartitions=1
    )
    y = dd.from_pandas(pd.Series(data["target"], name="TARGET"), npartitions=1)
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
        ["Estimator", Model(model)],
    ]
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X


def test_pipeline_fit_and_transform(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    X_new = pipe.transform(X)
    assert X_new.compute().shape == (150, 4)
    assert list(X_new.columns) == [
        "sepal length (cm)_new",
        "sepal width (cm)_new",
        "petal length (cm)_new",
        "petal width (cm)_new",
    ]


def test_fit_transform_pipeline(pipeline_example):
    pipe, X = pipeline_example
    X_new = pipe.fit_transform(X)
    assert X_new.compute().shape == (150, 4)
    assert list(X_new.columns) == [
        "sepal length (cm)_new",
        "sepal width (cm)_new",
        "petal length (cm)_new",
        "petal width (cm)_new",
    ]


def test_pipeline_predict(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict(X)
    assert y_pred.shape == (150,)


def test_pipeline_predict_proba(pipeline_with_model_example):
    pipe, X = pipeline_with_model_example
    y_pred = pipe.predict_proba(X)
    assert y_pred.shape == (150, 3)


def test_pipeline_numpy(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    X_numpy_new = pipe.transform_numpy(X.compute().to_numpy())
    assert X_numpy_new.shape == (150, 4)


def test_pipeline_transform_input_data(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    with pytest.raises(TypeError):
        _ = pipe.transform(X.compute().to_numpy())
    with pytest.raises(TypeError):
        _ = pipe.transform(X, X)
    with pytest.raises(TypeError):
        _ = pipe.transform_numpy(X)
