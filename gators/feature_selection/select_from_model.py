# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import pandas as pd

from ..converter import KoalasToPandas
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection


class SelectFromModel(_BaseFeatureSelection):
    """Select From Model Transformer.

    Select the top *k* features based on the feature importance
    of the given machine learning model.

    Parameters
    ----------
    model : model
        Machine learning model.
    k : int
        Number of features to keep.

    Examples
    ---------
    * fit & transform with `pandas`


    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier as RFC
    >>> from gators.feature_selection import SelectFromModel
    >>> X = pd.DataFrame(
    ... {'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> model = RFC(n_estimators=1, max_depth=2, random_state=0)
    >>> obj = SelectFromModel(model=model, k=2)
    >>> obj.fit_transform(X, y)
           A    C
    0  22.00  3.0
    1  38.00  1.0
    2  26.00  3.0
    3  35.00  1.0
    4  35.00  3.0
    5  28.11  3.0
    6  54.00  1.0
    7   2.00  3.0
    8  27.00  3.0
    9  14.00  2.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from pyspark.ml.classification import RandomForestClassifier as RFCSpark
    >>> from gators.feature_selection import SelectFromModel
    >>> X = ks.DataFrame(
    ... {'A': [22.0, 38.0, 26.0, 35.0, 35.0, 28.11, 54.0, 2.0, 27.0, 14.0],
    ... 'B': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
    ... 'C': [3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0]})
    >>> y = ks.Series([0, 1, 1, 1, 0, 0, 0, 0, 1, 1], name='TARGET')
    >>> model = RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    >>> obj = SelectFromModel(model=model, k=2)
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
    gators.feature_selection.SelectFromModels
        Similar method using multiple models.

    """

    def __init__(self, model, k: int):
        if not isinstance(k, int):
            raise TypeError("`k` should be an int.")
        if not hasattr(model, "fit"):
            raise TypeError("`model` should have the attribute `fit`.")
        _BaseFeatureSelection.__init__(self)
        self.model = model
        self.k = k

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "SelectFromModel":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
            SelectFromModel: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        columns = list(X.columns)
        if isinstance(X, pd.DataFrame):
            self.feature_importances_ = self.calculate_feature_importances_pd(
                model=self.model, X=X, y=y, columns=columns
            )
        else:
            if hasattr(self.model, "labelCol"):
                self.feature_importances_ = self.calculate_feature_importances_ks(
                    model=self.model, X=X, y=y, columns=columns
                )
            else:
                X_, y_ = KoalasToPandas().transform(X, y)
                self.feature_importances_ = self.calculate_feature_importances_pd(
                    model=self.model, X=X_, y=y_, columns=columns
                )
        mask = self.feature_importances_ != 0
        self.feature_importances_ = self.feature_importances_[mask]
        self.feature_importances_.sort_values(ascending=False, inplace=True)
        self.selected_columns = list(self.feature_importances_.index[: self.k])
        self.columns_to_drop = [c for c in columns if c not in self.selected_columns]
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self

    @staticmethod
    def calculate_feature_importances_pd(
        model: object, X: pd.DataFrame, y: Union[pd.Series, ks.Series], columns: list
    ) -> pd.Series:
        model.fit(X.to_numpy(), y)
        feature_importances_ = pd.Series(
            model.feature_importances_,
            index=columns,
        )
        return feature_importances_

    @staticmethod
    def calculate_feature_importances_ks(
        model: object, X: ks.DataFrame, y: ks.Series, columns: list
    ) -> pd.Series:
        spark_df = util.generate_spark_dataframe(X=X, y=y)
        trained_model = model.fit(spark_df)
        feature_importances_ = pd.Series(
            trained_model.featureImportances.toArray(), index=columns
        )
        return feature_importances_
