# License: Apache-2.0
from typing import Callable, Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin
from hyperopt.pyll.base import Apply
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


class HyperOpt:
    """Hyper parameter optimization using the Hyperopt package.

    Parameters
    ----------
    model : object
        Machine learning model.
    algo: Callable
        `algo` can be:

        * hp.rand.suggest for random search.
        * hp.tpe.suggest for tree of parzen estimators.
        * hp.atpe.suggest for adaptative tree of parzen estimators.

    scoring : _PredictScorer
        Score (or loss) function with signature score_func(y, y_pred).
    space : Dict[str, Apply]
            Hyperopt search space.
    kfold : Union[KFold, StratifiedKFold]
            sklearn K-Folds cross-validator.
    max_evals : int
        Number of evaluations.
    features: List[str]
        List of feature names.

    Examples
    --------

    >>> import numpy as np
    >>> import sklearn
    >>> from hyperopt import hp, tpe
    >>> from lightgbm import LGBMClassifier
    >>> from gators.model_building import HyperOpt
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y_np = np.array([0, 1, 1, 0])
    >>> hyper = HyperOpt(
    ...     model=LGBMClassifier(random_state=0),
    ...     algo=tpe.suggest,
    ...     scoring=sklearn.metrics.make_scorer(sklearn.metrics.f1_score),
    ...     space={'max_depth': hp.quniform('max_depth',1, 3, 1)},
    ...     max_evals=3,
    ...     kfold=StratifiedKFold(n_splits=2),
    ...     features=['x', 'y'],
    ... )
    >>> _ = hyper.fit(X_np, y_np)
    100%|##########| 3/3 [00:03<00:00,  1.03s/trial, best loss: -0.0]
    >>> hyper.model
    LGBMClassifier(max_depth=2, random_state=0)

    """

    def __init__(
        self,
        model: object,
        scoring: _PredictScorer,
        algo: Callable,
        space: Dict[str, Apply],
        kfold: Union[KFold, StratifiedKFold],
        max_evals: int,
        features: List[str],
    ) -> "HyperOpt":

        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("`model` should have the methods `fit` and `predict`")
        if not callable(algo):
            raise TypeError(
                """
                `algo` should be:
                hyperopt.rand, hyperopt.tpe, or hyperopt.atpe"""
            )
        if not isinstance(scoring, (str, _PredictScorer)):
            raise TypeError(
                """`scoring` should be a str or
                a sklearn.metrics._scorer._PredictScorer"""
            )
        if not isinstance(space, dict):
            raise TypeError("`space` should be a dict")
        if not isinstance(kfold, (KFold, StratifiedKFold)):
            raise TypeError(
                "`scoring` should be a sklearn.metrics._scorer._PredictScorer"
            )
        if not isinstance(max_evals, int):
            raise TypeError("`max_evals` should be an int")
        if not isinstance(features, list):
            raise TypeError("`features` should be a list")
        self.model = model
        self.algo = algo
        self.scoring = scoring
        self.space = space
        self.kfold = kfold
        self.max_evals = max_evals
        self.history = pd.DataFrame()
        self.int_parameters: List[str] = self.get_int_parameters(space)
        self.features = features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HyperOpt":
        """Fit model on X with y.

        Parameters
        ----------
        X : np.ndarray
           NumPy array.
        y : np.ndarray
            Target values.

        Returns
        -------
        HyperOpt
            Instance of itself.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("""`X` must be a NumPy array.""")
        if not isinstance(y, np.ndarray):
            raise TypeError("""`y` must be a NumPy array.""")

        def fn(params, cv=self.kfold, X=X, y=y):
            for int_parameter in self.int_parameters:
                params[int_parameter] = int(params[int_parameter])
            self.model.set_params(**params)
            self.model.fit(X, y)
            score = cross_val_score(
                self.model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1
            ).mean()
            return -score

        trials = Trials()
        best = fmin(
            fn=fn,
            space=self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.RandomState(0),
        )
        for int_parameter in self.int_parameters:
            best[int_parameter] = int(best[int_parameter])
        self.history = self.generate_history(trials)
        self.model.set_params(**best)
        self.model.fit(X, y)
        return self

    def get_feature_importances(self) -> pd.Series:
        """Generate the feature importances from the selected model.

        Returns
        -------
        pd.Series
            Feature importances.
        """
        feature_importances_ = pd.Series(self.model.feature_importances_, self.features)
        return feature_importances_.sort_values(ascending=False)

    @staticmethod
    def generate_history(trials: object) -> pd.DataFrame:
        """Generate hyperparameter tuning history.

        Parameters
        ----------
        trials : object
            Hyperopt trial funnction.

        Returns
        -------
        pd.DataFrame
            Hyperparameter tuning history.
        """

        def f(x) -> ks.Series[np.float32]:
            return pd.Series({key: val[0] for key, val in x.items()})

        loss = pd.DataFrame(trials.results)
        params = pd.DataFrame(trials.miscs)["vals"].apply(f)
        history = pd.concat([params, loss], axis=1)
        history["id"] = history.index
        history.sort_values("loss", ascending=False)
        return history

    @staticmethod
    def get_int_parameters(space) -> List[str]:
        """Get the list of int parameters based on the hyperopt search space.

        Parameters
        ----------
        space : object
            Hyperopt search space.

        Returns
        -------
        List[str]
            List of int parameters.
        """
        int_parameters = []
        for key in space.keys():
            if "qlog" in str(space[key]) or "quni" in str(space[key]):
                int_parameters.append(key)
        return int_parameters
