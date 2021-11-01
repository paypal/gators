# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import pandas as pd

from ..binning._base_discretizer import _BaseDiscretizer
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection
from .information_value import InformationValue


class MultiClassInformationValue(_BaseFeatureSelection):
    """Feature selection based on the information value.

    `MultiClassInformationValue` accepts only for muti-class variable targets.

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
    gators.feature_selection.RegressionInformationValue
        Information value for regression problems.

    Examples
    --------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_selection import MultiClassInformationValue
    >>> from gators.binning import Discretizer
    >>> X = pd.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> y = pd.Series([1, 1, 2, 2, 0, 0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = MultiClassInformationValue(k=3, discretizer=discretizer)
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
    >>> from gators.feature_selection import InformationValue
    >>> from gators.binning import Discretizer
    >>> X = ks.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> y = ks.Series([1, 1, 2, 2, 0, 0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = MultiClassInformationValue(k=3, discretizer=discretizer)
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
    >>> from gators.feature_selection import InformationValue
    >>> from gators.binning import Discretizer
    >>> X = pd.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> y = pd.Series([1, 1, 2, 2, 0, 0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = MultiClassInformationValue(k=3, discretizer=discretizer)
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
    >>> y = ks.Series([1, 1, 2, 2, 0, 0], name='TARGET')
    >>> discretizer = Discretizer(n_bins=4)
    >>> obj = MultiClassInformationValue(k=3, discretizer=discretizer)
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
    ) -> "MultiClassInformationValue":
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
        self.check_multiclass_target(y)
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
        discretizer.inplace = False
        if isinstance(X, pd.DataFrame):
            y_onehot = pd.get_dummies(y, prefix=y.name)
        else:
            y_onehot = ks.get_dummies(y, prefix=y.name)
        y_onehot = y_onehot.drop(y_onehot.columns[0], axis=1)
        information_values = pd.DataFrame(index=X.columns, columns=y_onehot.columns[1:])
        iv = InformationValue(discretizer=discretizer, k=X.shape[1])
        object_columns = util.get_datatype_columns(X, object)
        columns = object_columns + discretizer.output_columns
        for col in y_onehot.columns[1:]:
            _ = iv.fit(X[columns], y_onehot.loc[:, col])
            information_values.loc[:, col] = iv.feature_importances_
        return information_values.fillna(0).max(1).sort_values(ascending=False)
