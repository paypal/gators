# License: Apache-2.0
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gators.model_building.model import Model

from sklearn.datasets import load_iris

data_iris = load_iris()


@pytest.fixture()
def data():
    X = pd.DataFrame(data_iris["data"], columns=data_iris["feature_names"])
    y = pd.Series(data_iris["target"], name="TARGET")
    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0)
    obj = Model(model=model).fit(X, y)
    return obj, X, y


@pytest.mark.koalas
def test_predict(data):
    obj, X, y = data
    y_pred = obj.predict(X)
    assert y_pred.shape == y.shape


@pytest.mark.koalas
def test_predict_proba(data):
    obj, X, y = data
    y_pred_proba = obj.predict_proba(X)
    assert y_pred_proba.shape[0] == y.shape[0]
    assert y_pred_proba.shape[1] == 3
