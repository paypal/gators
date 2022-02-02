# License: Apache-2.0
from typing import List

import numpy as np
import pandas as pd

from ..scalers.minmax_scaler import MinMaxScaler
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

from gators import DataFrame, Series


class SelectFromModels(_BaseFeatureSelection):
    """Select From Models By Vote Transformer.

    Select the top *k* features based on the feature importance
    of the given machine learning models.

    Parameters
    ----------
    models : List[model]
        List of machine learning models.
    k : int
        Number of features to keep.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_selection import SelectFromModels

    Note that the model can be:

          * a **xgboost.dask** or a **sklearn** model for `dask` dataframes
          * a **sklearn** model for `pandas` and `pandas` dataframes
          * a **pyspark.ml** model for `koalas` dataframes

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> from xgboost.dask import XGBClassifier
    >>> from distributed import Client, LocalCluster
    >>> cluster = LocalCluster()
    >>> client = Client(cluster)
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [0.94, 0.09, -0.43, 0.31, 0.99, 1.05, 1.02, -0.77, 0.03, 0.99],
    ... 'B': [0.13, 0.01, -0.06, 0.04, 0.14, 0.14, 0.14, -0.1, 0.0, 0.13],
    ... 'C': [0.8, 0.08, -0.37, 0.26, 0.85, 0.9, 0.87, -0.65, 0.02, 0.84]}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([1, 0, 0, 0, 1, 1, 1, 0, 0, 1], name='TARGET'), npartitions=1)
    >>> models = [
    ... XGBClassifier(n_estimators=1, random_state=0, eval_metric='logloss', use_label_encoder=False),
    ... XGBClassifier(n_estimators=1, random_state=1, eval_metric='logloss', use_label_encoder=False)]
    >>> models[0].client = client
    >>> models[1].client = client
    >>> obj = SelectFromModels(models=models, k=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> from pyspark import SparkConf, SparkContext
    >>> from pyspark.ml.classification import RandomForestClassifier as RFCSpark
    >>> conf = SparkConf()
    >>> _ = conf.set('spark.executor.memory', '2g')
    >>> _ = SparkContext(conf=conf)
    >>> X = ks.DataFrame({
    ... 'A': [0.94, 0.09, -0.43, 0.31, 0.99, 1.05, 1.02, -0.77, 0.03, 0.99],
    ... 'B': [0.13, 0.01, -0.06, 0.04, 0.14, 0.14, 0.14, -0.1, 0.0, 0.13],
    ... 'C': [0.8, 0.08, -0.37, 0.26, 0.85, 0.9, 0.87, -0.65, 0.02, 0.84]})
    >>> y = ks.Series([1, 0, 0, 0, 1, 1, 1, 0, 0, 1], name='TARGET')
    >>> models = [RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0),
    ... RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=1)]
    >>> obj = SelectFromModels(models=models, k=1)


    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> from xgboost import XGBClassifier
    >>> X = pd.DataFrame({
    ... 'A': [0.94, 0.09, -0.43, 0.31, 0.99, 1.05, 1.02, -0.77, 0.03, 0.99],
    ... 'B': [0.13, 0.01, -0.06, 0.04, 0.14, 0.14, 0.14, -0.1, 0.0, 0.13],
    ... 'C': [0.8, 0.08, -0.37, 0.26, 0.85, 0.9, 0.87, -0.65, 0.02, 0.84]})
    >>> y = pd.Series([1, 0, 0, 0, 1, 1, 1, 0, 0, 1], name='TARGET')
    >>> models = [XGBClassifier(n_estimators=1, max_depth=3, random_state=0, eval_metric='logloss'),
    ... XGBClassifier(n_estimators=1, max_depth=4, random_state=1, eval_metric='logloss')]
    >>> obj = SelectFromModels(models=models, k=1)

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X, y)
          A
    0  0.94
    1  0.09
    2 -0.43
    3  0.31
    4  0.99
    5  1.05
    6  1.02
    7 -0.77
    8  0.03
    9  0.99

    See Also
    --------
    gators.feature_selection.SelectFromMode
        Similar method using one model.

    """

    def __init__(self, models: List[object], k: int):
        if not isinstance(models, (list, np.ndarray)):
            raise TypeError("`models` should be a list.")
        if not isinstance(k, int):
            raise TypeError("`k` should be an int.")
        for model in models:
            if not hasattr(model, "fit"):
                raise TypeError(
                    "All the elements of `models` should have the attribute `fit`."
                )
        _BaseFeatureSelection.__init__(self)
        self.models = models
        self.k = k

    def fit(self, X: DataFrame, y: Series = None) -> "SelectFromModels":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : "SelectFromModels"
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        self.feature_importances_ = self.get_feature_importances_frame(X, self.models)
        for col, model in zip(self.feature_importances_.columns, self.models):
            self.feature_importances_[col] = util.get_function(X).feature_importances_(
                model, X, y
            )
        self.feature_importances_ = self.clean_feature_importances_frame(
            self.feature_importances_
        )
        self.selected_columns = list(
            self.feature_importances_["count"].iloc[: self.k].index
        )
        self.columns_to_drop = [
            c for c in self.feature_importances_.index if c not in self.selected_columns
        ]
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self

    @staticmethod
    def get_feature_importances_frame(X, models):
        index = np.array(list(X.columns))
        columns = []
        for i, model in enumerate(models):
            col = str(model).split("(")[0]
            columns.append(col + "_" + str(i))
        return pd.DataFrame(columns=columns, index=index, dtype=np.float64)

    @staticmethod
    def clean_feature_importances_frame(feature_importances):
        feature_importances = MinMaxScaler().fit_transform(feature_importances)
        feature_importances_sum = feature_importances.sum(1)
        feature_importances_count = (feature_importances != 0).sum(1)
        feature_importances["sum"] = feature_importances_sum
        feature_importances["count"] = feature_importances_count
        feature_importances.sort_values(
            by=["count", "sum"], ascending=False, inplace=True
        )
        return feature_importances
