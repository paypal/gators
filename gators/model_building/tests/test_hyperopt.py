# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from hyperopt import hp, tpe
from pandas.testing import assert_series_equal
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from gators.model_building.hyperopt import HyperOpt


@pytest.fixture
def data():
    X, y = make_classification(
        random_state=0, n_samples=10, n_features=5, n_informative=3
    )
    model = XGBClassifier(random_state=0, eval_metric="logloss", max_depth=2)
    n_splits = 2
    max_evals = 10
    kfold = StratifiedKFold(n_splits=n_splits)
    space = {"n_estimators": hp.quniform("n_estimators", 5, 25, 5)}
    model = XGBClassifier(
        random_state=0, eval_metric="logloss", use_label_encoder=False
    )

    def f1_score(y_true, y_pred):
        p = y_true[y_pred == 1].mean()
        r = y_pred[y_true == 1].mean()
        if (p == 0) | (r == 0):
            return 0
        return 2 * p * r / (p + r)

    f1_scoring = make_scorer(f1_score)
    y_pred_expected = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    return model, f1_scoring, space, max_evals, kfold, X, y, y_pred_expected


def test_hyperopt(data):
    model, f1_scoring, space, max_evals, kfold, X, y, y_pred_expected = data
    hyper = HyperOpt(
        model=model,
        algo=tpe.suggest,
        scoring=f1_scoring,
        space=space,
        max_evals=max_evals,
        kfold=kfold,
        features=["A", "B", "C", "D", "E"],
    ).fit(X, y)
    y_pred = hyper.model.predict(X)
    feature_importances = hyper.get_feature_importances()
    assert feature_importances[-3:].sum() == 0
    assert np.allclose(y_pred, y_pred_expected)


def test_input(data):
    model, f1_scoring, space, max_evals, kfold, X, y, y_pred_expected = data
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=0,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=space,
            max_evals=max_evals,
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=0,
            scoring=f1_scoring,
            space=space,
            max_evals=max_evals,
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=0,
            space=space,
            max_evals=max_evals,
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=0,
            max_evals=max_evals,
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=space,
            max_evals="a",
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=space,
            max_evals=max_evals,
            kfold=0,
            features=["A", "B", "C", "D", "E"],
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=space,
            max_evals=max_evals,
            kfold=kfold,
            features=0,
        )
    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=space,
            max_evals=max_evals,
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        ).fit(0, 0)

    with pytest.raises(TypeError):
        _ = HyperOpt(
            model=model,
            algo=tpe.suggest,
            scoring=f1_scoring,
            space=space,
            max_evals=max_evals,
            kfold=kfold,
            features=["A", "B", "C", "D", "E"],
        ).fit(X, 0)
