# License: Apache-2.0
import databricks.koalas as ks
import pytest
from pyspark.ml.classification import RandomForestClassifier as RFCSpark

from gators.model_building.model import Model

ks.set_option("compute.default_index_type", "distributed-sequence")

from sklearn.datasets import load_iris

data_iris = load_iris()


@pytest.fixture()
def data():
    X = ks.DataFrame(data_iris["data"], columns=data_iris["feature_names"])
    y = ks.Series(data_iris["target"], name="TARGET")
    model = RFCSpark(numTrees=5, maxDepth=2, labelCol=y.name, seed=0)
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
    assert y_pred_proba.shape == y.shape
