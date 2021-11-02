# License: Apache-2.0
from typing import Union

import numpy as np
import xgboost
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor


class XGBBoosterBuilder:
    """XGBoost Booster Converter Class.

    Examples
    --------
    >>> import numpy as np
    >>> import xgboost as xgb
    >>> from gators.model_building import XGBBoosterBuilder
    >>> X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y_train = np.array([0, 1, 1, 0])
    >>> model = xgb.XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
    >>> xgbooster = XGBBoosterBuilder.train(
    ... model=model,
    ... X_train=X_train,
    ... y_train=y_train)
    >>> xgbooster.predict(xgb.DMatrix(X_train))
    array([0.5, 0.5, 0.5, 0.5], dtype=float32)

    """

    @staticmethod
    def train(
        model: Union[XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor],
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_class=None,
    ):
        """Convert the XGBoost model to a XGB Booster model.

        Parameters
        ----------
        model : Union[XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor]
            Trained xgboost.sklearn model.
        X_train : np.ndarray
            Train array.
        y_train : np.ndarray
             Target values.

        Returns
        -------
        xgboost.Booster
            Trained xgboost Booster model.
        """
        if not isinstance(
            model, (XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor)
        ):
            raise TypeError("`model` should be a `xgboost.sklearn` model.")
        if not isinstance(X_train, np.ndarray):
            raise TypeError("`X_train` should be a NumPy array.")
        if not isinstance(y_train, np.ndarray):
            raise TypeError("`y_train` should be a NumPy array.")
        if num_class is not None and not isinstance(num_class, int):
            raise TypeError("`num_class` should be an int.")
        params = model.get_xgb_params()
        if num_class:
            params["num_class"] = num_class
        n_estimators = model.n_estimators
        dtrain = xgboost.DMatrix(X_train, label=y_train)
        return xgboost.train(params, dtrain, n_estimators)
