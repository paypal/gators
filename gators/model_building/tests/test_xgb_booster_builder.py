# License: Apache-2.0
import numpy as np
import pytest
import xgboost
from xgboost import XGBClassifier

from gators.model_building.xgb_booster_builder import XGBBoosterBuilder


def test():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    xgb = XGBClassifier(n_estimators=2, max_depth=2, use_label_encoder=False)
    xgb.fit(X_train, y_train)
    xgb_booster = XGBBoosterBuilder.train(xgb, X_train, y_train)
    assert np.allclose(
        xgb.predict_proba(X_train)[:, 1], xgb_booster.predict(xgboost.DMatrix(X_train))
    )


def test_num_class():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 2, 0])
    xgb = XGBClassifier(n_estimators=2, max_depth=2)
    xgb.fit(X_train, y_train)
    xgb_booster = XGBBoosterBuilder.train(xgb, X_train, y_train, num_class=3)
    assert np.allclose(
        xgb.predict_proba(X_train), xgb_booster.predict(xgboost.DMatrix(X_train))
    )


def test_input():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    xgb = XGBClassifier(n_estimators=2, max_depth=2)
    xgb.fit(X_train, y_train)
    num_class = 2
    with pytest.raises(TypeError):
        _ = XGBBoosterBuilder.train(0, X_train, y_train, num_class)
    with pytest.raises(TypeError):
        _ = XGBBoosterBuilder.train(xgb, 0, y_train, num_class)
    with pytest.raises(TypeError):
        _ = XGBBoosterBuilder.train(xgb, X_train, 0, num_class)
    with pytest.raises(TypeError):
        _ = XGBBoosterBuilder.train(xgb, X_train, y_train, "a")
