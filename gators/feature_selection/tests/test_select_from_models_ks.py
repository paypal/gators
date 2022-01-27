# License: Apache-2.0
import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyspark.ml.classification import RandomForestClassifier as RFCSpark

from gators.feature_selection.select_from_models import SelectFromModels

ks.set_option("compute.default_index_type", "distributed-sequence")


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
            "B": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name="TARGET")
    X_expected = X[["A", "B"]].to_pandas().copy()
    model1_ks = RFCSpark(numTrees=2, maxDepth=2, labelCol=y.name, seed=0)
    mode2_ks = RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    obj = SelectFromModels(models=[model1_ks, mode2_ks], k=2).fit(X, y)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(np.float64))
