# License: Apache-2.0
from sklearn.base import BaseEstimator


from ..util import util
from gators import DataFrame, Series


class Model(BaseEstimator):
    """Model wrapper class.

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

    def __init__(self, model):
        """Wrap the model in the SKLearn fit-transform method.

        Parameters
        ----------
        model : "PySparkML model"
            model.
        """
        self.model = model

    def fit(self, X: DataFrame, y: Series) -> "Model":
        """Fit the model on X and y.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series
            Target values

        Returns
        -------
        "SparkMLWrapper"
            Instance of itself.
        """
        self.model = util.get_function(X).fit(self.model, X, y)
        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        DataFrame
            The predicted classes of the input dataframe.
        """
        return util.get_function(X).predict(self.model, X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        DataFrame
            The predicted class probabilities of the input dataframe.
        """
        return util.get_function(X).predict_proba(self.model, X)
