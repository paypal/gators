# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from xgboost import XGBClassifier

from gators.feature_selection.select_from_model import SelectFromModel


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name="TARGET")
    X_expected = X[["A"]].copy()
    model = XGBClassifier(
        random_state=0,
        subsample=1.0,
        n_estimators=2,
        max_depth=2,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    obj = SelectFromModel(model=model, k=1).fit(X, y)
    return obj, X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new)
    assert_frame_equal(X_new, pd.DataFrame(X_expected.to_numpy()))


def test_init():
    with pytest.raises(TypeError):
        _ = SelectFromModel(model=XGBClassifier(), k="a")

    class Model:
        pass

    with pytest.raises(TypeError):
        _ = SelectFromModel(model=Model(), k=2)
