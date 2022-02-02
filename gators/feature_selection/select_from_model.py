# License: Apache-2.0


from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

from gators import DataFrame, Series


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
    Imports and initialization:

    >>> from gators.feature_selection import SelectFromModel

    Note that the model can be:

          * a **xgboost.dask** or a **sklearn** model for `dask` dataframes
          * a **sklearn** model for `pandas` and `dask` dataframes
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
    >>> model = XGBClassifier(
    ... n_estimators=1, random_state=0, eval_metric='logloss', use_label_encoder=False)
    >>> model.client = client
    >>> obj = SelectFromModel(model=model, k=1)

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
    >>> model = RFCSpark(numTrees=1, maxDepth=2, labelCol=y.name, seed=0)
    >>> obj = SelectFromModel(model=model, k=1)

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> from xgboost import XGBClassifier
    >>> X = pd.DataFrame({
    ... 'A': [0.94, 0.09, -0.43, 0.31, 0.99, 1.05, 1.02, -0.77, 0.03, 0.99],
    ... 'B': [0.13, 0.01, -0.06, 0.04, 0.14, 0.14, 0.14, -0.1, 0.0, 0.13],
    ... 'C': [0.8, 0.08, -0.37, 0.26, 0.85, 0.9, 0.87, -0.65, 0.02, 0.84]})
    >>> y = pd.Series([1, 0, 0, 0, 1, 1, 1, 0, 0, 1], name='TARGET')
    >>> model = XGBClassifier(n_estimators=1, random_state=0, eval_metric='logloss')
    >>> obj = SelectFromModel(model=model, k=1)

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

    def fit(self, X: DataFrame, y: Series = None) -> "SelectFromModel":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : "SelectFromModel"
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        columns = list(X.columns)
        self.feature_importances_ = util.get_function(X).feature_importances_(
            self.model, X, y
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
