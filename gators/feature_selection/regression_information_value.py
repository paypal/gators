# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import pandas as pd

from ..binning._base_discretizer import _BaseDiscretizer
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection
from .multiclass_information_value import MultiClassInformationValue


class RegressionInformationValue(_BaseFeatureSelection):
    """Regression Information Value Transformer.

    `RegressionInformationValue` accepts only continuous variable targets.

    Parameters
    ----------
    k : int
        Number of features to keep.
    discretizer : _BaseDiscretizer
        Discretizer Transformer.

    See Also
    --------
    gators.feature_selection.InformationValue
        Information value for binary classification problems.
    gators.feature_selection.MultiClassInformationValue
        Information value for multi-class classification problems.

    Examples
    --------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_selection import RegressionInformationValue
    >>> from gators.binning import Discretizer
    >>> X = pd.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> y = pd.Series([11.56, 9.57, 33.33, 87.6, 0.01, -65.0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = RegressionInformationValue(k=3, discretizer=discretizer)
    >>> obj.fit_transform(X, y)
           A  B  C
    0  87.25  1  a
    1   5.25  1  b
    2  70.25  0  b
    3   5.25  1  b
    4   0.25  0  a
    5   7.25  0  a

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.binning import Discretizer
    >>> from gators.feature_selection import InformationValue
    >>> X = ks.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> X_expected = X[['A', 'B', 'C']].copy()
    >>> y = ks.Series([11.56, 9.57, 33.33, 87.6, 0.01, -65.0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = RegressionInformationValue(k=3, discretizer=discretizer)
    >>> obj.fit_transform(X, y)
           A  B  C
    0  87.25  1  a
    1   5.25  1  b
    2  70.25  0  b
    3   5.25  1  b
    4   0.25  0  a
    5   7.25  0  a

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_selection import RegressionInformationValue
    >>> from gators.binning import Discretizer
    >>> X = pd.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> X_expected = X[['A', 'B', 'C']].copy()
    >>> y = pd.Series([11.56, 9.57, 33.33, 87.6, 0.01, -65.0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = RegressionInformationValue(k=3, discretizer=discretizer)
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[87.25, 1, 'a'],
           [5.25, 1, 'b'],
           [70.25, 0, 'b'],
           [5.25, 1, 'b'],
           [0.25, 0, 'a'],
           [7.25, 0, 'a']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_selection import InformationValue
    >>> from gators.binning import Discretizer
    >>> X = ks.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> y = ks.Series([11.56, 9.57, 33.33, 87.6, 0.01, -65.0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = RegressionInformationValue(k=3, discretizer=discretizer)
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[87.25, 1, 'a'],
           [5.25, 1, 'b'],
           [70.25, 0, 'b'],
           [5.25, 1, 'b'],
           [0.25, 0, 'a'],
           [7.25, 0, 'a']], dtype=object)


    """

    def __init__(self, k: int, discretizer: _BaseDiscretizer):
        if not isinstance(k, int):
            raise TypeError("`k` should be an int.")
        if not isinstance(discretizer, _BaseDiscretizer):
            raise TypeError("`discretizer` should inherite from _BaseDiscretizer.")
        _BaseFeatureSelection.__init__(self)
        self.k = k
        self.discretizer = discretizer

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "RegressionInformationValue":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
            InformationValue: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.check_regression_target(y)
        columns = X.columns
        self.feature_importances_ = self.compute_information_value(
            X, y, self.discretizer
        )
        self.feature_importances_.sort_values(ascending=False, inplace=True)
        self.selected_columns = list(self.feature_importances_.index[: self.k])
        self.columns_to_drop = [c for c in columns if c not in self.selected_columns]
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self

    @staticmethod
    def compute_information_value(
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series],
        discretizer: _BaseDiscretizer,
    ) -> pd.Series:
        """Compute information value.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.
        discretizer : _BaseDiscretizer
            Discretizer Transformer.

        Returns
        -------
        pd.Series
            Information value.
        """
        discretizer.inplace = True
        y_binned = discretizer.fit_transform(y.to_frame())[y.name]
        discretizer.inplace = False
        discretizer.output_columns = []
        return MultiClassInformationValue.compute_information_value(
            X, y_binned.astype(float).astype(int), discretizer=discretizer
        )
