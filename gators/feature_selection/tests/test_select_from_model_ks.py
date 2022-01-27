# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pyspark.ml.classification import RandomForestClassifier as RFCSpark

from gators.feature_selection.select_from_model import SelectFromModel

ks.set_option("compute.default_index_type", "distributed-sequence")


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
    X_expected = X[["A"]].copy()
    model = RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    obj = SelectFromModel(model=model, k=2).fit(X, y)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    X_new = X_new.to_pandas()
    assert X_expected.shape == X_new.shape


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert X_expected.shape == X_new.shape
