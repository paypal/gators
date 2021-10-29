# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.ensemble import RandomForestClassifier

from gators.feature_selection.select_from_model import SelectFromModel
from gators.pipeline.pipeline import Pipeline
from gators.transformers.transformer import Transformer


class MultiplyTransformer(Transformer):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        return self.multiplier * X

    def transform_numpy(self, X):
        return self.multiplier * X


class NameTransformer(Transformer):
    def fit(self, X, y=None):
        self.column_names = [f"{c}_new" for c in X.columns]
        self.column_mapping = dict(zip(self.column_names, [[c] for c in X.columns]))
        self.column_mapping["D_new"] = "D"
        return self

    def transform(self, X):
        return X.rename(columns=dict(zip(X.columns, self.column_names)))

    def transform_numpy(self, X):
        return X


@pytest.fixture
def pipeline_example():
    X = pd.DataFrame(
        [
            [1.764, 0.4, 0.979, 2.241],
            [1.868, -0.977, 0.95, -0.151],
            [-0.103, 0.411, 0.144, 1.454],
            [0.761, 0.122, 0.444, 0.334],
        ],
        columns=list("ABCD"),
    )
    y = pd.Series([0, 1, 0, 1], name="TARGET")
    steps = [
        MultiplyTransformer(4.0),
        MultiplyTransformer(0.5),
        NameTransformer(),
    ]
    pipe = Pipeline(steps)
    X_expected = pd.DataFrame(
        {
            "A_new": {0: 3.528, 1: 3.736, 2: -0.206, 3: 1.522},
            "B_new": {0: 0.8, 1: -1.954, 2: 0.822, 3: 0.244},
            "C_new": {0: 1.958, 1: 1.9, 2: 0.288, 3: 0.888},
            "D_new": {0: 4.482, 1: -0.302, 2: 2.908, 3: 0.668},
        }
    )
    return pipe, X, X_expected


@pytest.fixture
def pipeline_with_feature_selection_example():
    X = pd.DataFrame(
        [
            [1.764, 0.4, 0.979, 2.241],
            [1.868, -0.977, 0.95, -0.151],
            [-0.103, 0.411, 0.144, 1.454],
            [0.761, 0.122, 0.444, 0.334],
        ],
        columns=list("ABCD"),
    )
    y = pd.Series([0, 1, 0, 1], name="TARGET")
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        MultiplyTransformer(4.0),
        MultiplyTransformer(0.5),
        NameTransformer(),
        SelectFromModel(model=model, k=3),
    ]
    pipe = Pipeline(steps).fit(X, y)
    X_expected = pd.DataFrame(
        {
            "A_new": {0: 3.528, 1: 3.736, 2: -0.206, 3: 1.522},
            "B_new": {0: 0.8, 1: -1.954, 2: 0.822, 3: 0.244},
            "C_new": {0: 1.958, 1: 1.9, 2: 0.288, 3: 0.888},
            "D_new": {0: 4.482, 1: -0.302, 2: 2.908, 3: 0.668},
        }
    )
    return pipe, X, X_expected


@pytest.fixture
def pipeline_with_model_example():
    X = pd.DataFrame(
        [
            [1.764, 0.4, 0.979, 2.241],
            [1.868, -0.977, 0.95, -0.151],
            [-0.103, 0.411, 0.144, 1.454],
            [0.761, 0.122, 0.444, 0.334],
        ],
        columns=list("ABCD"),
    )
    y = pd.Series([0, 1, 0, 1], name="TARGET")
    model = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0)
    steps = [
        MultiplyTransformer(4.0),
        MultiplyTransformer(0.5),
        NameTransformer(),
        model,
    ]
    X_expected = pd.DataFrame(
        {
            "A_new": {0: 3.528, 1: 3.736, 2: -0.206, 3: 1.522},
            "B_new": {0: 0.8, 1: -1.954, 2: 0.822, 3: 0.244},
            "C_new": {0: 1.958, 1: 1.9, 2: 0.288, 3: 0.888},
            "D_new": {0: 4.482, 1: -0.302, 2: 2.908, 3: 0.668},
        }
    )
    pipe = Pipeline(steps).fit(X, y)
    return pipe, X, X_expected


def test_pandas_pipeline_fit_and_transform(pipeline_example):
    pipe, X, X_expected = pipeline_example
    _ = pipe.fit(X)
    X_new = pipe.transform(X)
    assert_frame_equal(X_expected, X_new)


def test_pandas_fit_transform_pipeline(pipeline_example):
    pipe, X, X_expected = pipeline_example
    X_new = pipe.fit_transform(X)
    assert_frame_equal(X_expected, X_new)


def test_pipeline_predict_pandas(pipeline_with_model_example):
    pipe, X, X_expected = pipeline_with_model_example
    y_pred = pipe.predict(X)
    assert y_pred.shape == (4,)


def test_pipeline_predict_proba_pandas(pipeline_with_model_example):
    pipe, X, X_expected = pipeline_with_model_example
    y_pred = pipe.predict_proba(X)
    assert y_pred.shape == (4, 2)


def test_pipeline_numpy(pipeline_example):
    pipe, X, X_expected = pipeline_example
    _ = pipe.fit(X)
    X_numpy_new = pipe.transform_numpy(X.to_numpy())
    assert np.allclose(X_expected.to_numpy(), X_numpy_new)


def test_pipeline_predict_numpy(pipeline_with_model_example):
    pipe, X, X_expected = pipeline_with_model_example
    y_pred = pipe.predict_numpy(X.to_numpy())
    assert y_pred.shape == (4,)


def test_pipeline_predict_proba_numpy(pipeline_with_model_example):
    pipe, X, X_expected = pipeline_with_model_example
    y_pred = pipe.predict_proba_numpy(X.to_numpy())
    assert y_pred.shape == (4, 2)


def test_default_fit_transform_pipeline(pipeline_example):
    pipe, X, X_expected = pipeline_example
    X_new = pipe.fit_transform(X)
    assert_frame_equal(X_expected, X_new)


def test_init():
    with pytest.raises(TypeError):
        _ = Pipeline(0)
    with pytest.raises(TypeError):
        _ = Pipeline([])


def test_pipeline_transform_input_data(pipeline_example):
    pipe, X, _ = pipeline_example
    _ = pipe.fit(X)
    with pytest.raises(TypeError):
        _ = pipe.transform(X.to_numpy())
    with pytest.raises(TypeError):
        _ = pipe.transform(X, X)
    with pytest.raises(TypeError):
        _ = pipe.transform_numpy(X)


def test_get_feature_importances(pipeline_with_feature_selection_example):
    pipe, _, _ = pipeline_with_feature_selection_example
    feature_importances_expected = pd.Series({"D_new": 0.6, "B_new": 0.4})
    feature_importances = pipe.get_feature_importances(k=2)
    assert_series_equal(feature_importances, feature_importances_expected)


def test_get_features(pipeline_with_feature_selection_example):
    pipe, _, _ = pipeline_with_feature_selection_example
    assert ["D_new", "B_new"] == pipe.get_features()


def test_get_feature_importances_no_feature_selection(pipeline_example):
    pipe, _, _ = pipeline_example
    with pytest.raises(AttributeError):
        pipe.get_feature_importances(k=2)


def test_get_features_no_feature_selection(pipeline_example):
    pipe, _, _ = pipeline_example
    with pytest.raises(AttributeError):
        pipe.get_features()


def test_get_production_columns(pipeline_with_feature_selection_example):
    pipe, _, _ = pipeline_with_feature_selection_example
    assert pipe.get_production_columns() == ["B", "D"]


def test_get_production_columns_no_feature_selection(pipeline_example):
    pipe, _, _ = pipeline_example
    with pytest.raises(AttributeError):
        pipe.get_production_columns()
