# License: Apache-2.0import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from gators.encoders.woe_encoder import WOEEncoder
from gators.binning.binning import Binning
from gators.model_building.model import Model
from gators.feature_selection.select_from_model import SelectFromModel
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
        Transformer.check_dataframe(X)
        return X.rename(columns=dict(zip(X.columns, self.column_names)))

    def transform_numpy(self, X):
        Transformer.check_array(X)
        return X


@pytest.fixture
def pipeline_example():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
    ]
    pipe = Pipeline(steps)
    return pipe, X


@pytest.fixture
def pipeline_with_feature_selection_example():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")

    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
        ["SelectFromModel", SelectFromModel(model=model, k=3)],
    ]
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X


@pytest.fixture
def pipeline_with_model_example():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
        ["Estimator", Model(model)],
    ]
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X


def test_display_encoder_mapping_inplace():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")
    y = (y == 0).astype(int)
    steps = [
        ["Binning", Binning(n_bins=4)],
        ["Encoder", WOEEncoder()],
    ]
    pipe = Pipeline(steps).fit(X, y)
    pipe.display_encoder_mapping(cmap="Reds")
    pipe.display_encoder_mapping(cmap="Reds", decimals=0)
    assert True


def test_display_encoder_mapping():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")
    y = (y == 0).astype(int)
    steps = [
        ["Binning", Binning(n_bins=4, inplace=False)],
        ["Encoder", WOEEncoder()],
    ]
    pipe = Pipeline(steps).fit(X, y)
    pipe.display_encoder_mapping(cmap="Reds")
    assert True


def test_display_encoder_mapping_no_encoding():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")
    y = (y == 0).astype(int)
    steps = [
        ["Binning", Binning(n_bins=4, inplace=False)],
    ]
    pipe = Pipeline(steps).fit(X, y)
    pipe.display_encoder_mapping(cmap="Reds")
    assert True


def test_display_encoder_mapping_no_binning():
    X = pd.DataFrame(data["data"], columns=data["feature_names"]).astype(str)
    y = pd.Series(data["target"], name="TARGET")
    y = (y == 0).astype(int)
    steps = [
        ["Encoder", WOEEncoder()],
    ]
    pipe = Pipeline(steps).fit(X, y)
    pipe.display_encoder_mapping(cmap="Reds")
    assert True


def test_pipeline_fit_and_transform(pipeline_example):
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


def test_fit_transform_pipeline(pipeline_example):
    pipe, X = pipeline_example
    X_new = pipe.fit_transform(X)
    assert X_new.shape == (150, 4)
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
    X_numpy_new = pipe.transform_numpy(X.to_numpy())
    assert X_numpy_new.shape == (150, 4)


def test_init():
    with pytest.raises(TypeError):
        _ = Pipeline(0)
    with pytest.raises(TypeError):
        _ = Pipeline([])
    with pytest.raises(TypeError):
        _ = Pipeline(verbose=5)


def test_pipeline_transform_input_data(pipeline_example):
    pipe, X = pipeline_example
    _ = pipe.fit(X)
    with pytest.raises(TypeError):
        _ = pipe.transform(X.to_numpy())
    with pytest.raises(TypeError):
        _ = pipe.transform(X, X)
    with pytest.raises(TypeError):
        _ = pipe.transform_numpy(X)


def test_verbose():
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="TARGET")

    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        ["MultiplyTransformer1", MultiplyTransformer(4.0)],
        ["MultiplyTransformer2", MultiplyTransformer(0.5)],
        ["NameTransformer", NameTransformer()],
        ["Estimator", SelectFromModel(model=model, k=3)],
    ]
    pipe = Pipeline(steps, verbose=True).fit(X, y)
    _ = pipe.transform(X)
    _ = Pipeline(steps, verbose=True).fit_transform(X, y)
