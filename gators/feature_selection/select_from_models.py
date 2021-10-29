# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
import pyspark.sql.dataframe as ps

from ..scalers.minmax_scaler import MinMaxScaler
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection


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
    * fit & transform with `koalas`

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier as RFC
    >>> from gators.feature_selection import SelectFromModels
    >>> X = pd.DataFrame({
    ... 'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> models = [RFC(n_estimators=1, max_depth=1, random_state=0),
    ... RFC(n_estimators=1, max_depth=2, random_state=1)]
    >>> obj = SelectFromModels(models=models, k=2)
    >>> obj.fit_transform(X, y)
           B    C
    0   7.25  3.0
    1  71.28  1.0
    2   7.92  3.0
    3  53.10  1.0
    4   8.05  3.0
    5   8.46  3.0
    6  51.86  1.0
    7  21.08  3.0
    8  11.13  3.0
    9  30.07  2.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from pyspark.ml.classification import RandomForestClassifier as RFCSpark
    >>> from gators.feature_selection import SelectFromModels
    >>> X = ks.DataFrame({
    ... 'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> models = [RFCSpark(numTrees=1, maxDepth=1, labelCol=y.name, seed=0),
    ... RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=1)]
    >>> obj = SelectFromModels(models=models, k=2)
    >>> obj.fit_transform(X, y)
           A      B
    0  22.00   7.25
    1  38.00  71.28
    2  26.00   7.92
    3  35.00  53.10
    4  35.00   8.05
    5  28.11   8.46
    6  54.00  51.86
    7   2.00  21.08
    8  27.00  11.13
    9  14.00  30.07

    See Also
    --------
    gators.feature_selection.SelectFromMode
        Similar method using one model.

    """

    def __init__(self, models: List[object], k: int):
        if not isinstance(models, list):
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

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "SelectFromModels":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        SelectFromModels: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.feature_importances_ = self.get_feature_importances_frame(X, self.models)
        if isinstance(X, pd.DataFrame):
            for col, model in zip(self.feature_importances_.columns, self.models):
                model_feature_importances_ = self.get_feature_importances_pd(
                    model=model, X=X, y=y
                )
                self.feature_importances_[col] = model_feature_importances_
        else:
            spark_df = util.generate_spark_dataframe(X=X, y=y)
            for col, model in zip(self.feature_importances_.columns, self.models):
                model_feature_importances_ = self.get_feature_importances_sk(
                    model=model, spark_df=spark_df
                )
                self.feature_importances_[col] = model_feature_importances_
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
    def get_feature_importances_pd(
        model: object, X: pd.DataFrame, y: Union[pd.Series, ks.Series]
    ):
        model.fit(X, y)
        feature_importances_ = model.feature_importances_
        return feature_importances_

    @staticmethod
    def get_feature_importances_sk(model: object, spark_df: ps.DataFrame):
        trained_model = model.fit(spark_df)
        feature_importances_ = trained_model.featureImportances.toArray()
        return feature_importances_

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
