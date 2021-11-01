# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..binning._base_discretizer import _BaseDiscretizer
from ..util import util
from ._base_feature_selection import _BaseFeatureSelection


class InformationValue(_BaseFeatureSelection):
    """Feature selection based on the information value.

    `InformationValue` accepts only binary variable targets.

    Parameters
    ----------
    k : int
        Number of features to keep.
    discretizer : _BaseDiscretizer
        Discretizer transformer.

    See Also
    --------
    gators.feature_selection.MultiClassInformationValue
        Information value for muti-class classification problems.
    gators.feature_selection.RegressionInformationValue
        Information value for regression problems.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_selection import InformationValue
    >>> from gators.binning import Discretizer
    >>> X = pd.DataFrame({
    ...         'A': [87.25, 5.25, 70.25, 5.25, 0.25, 7.25],
    ...         'B': [1, 1, 0, 1, 0, 0],
    ...         'C': ['a', 'b', 'b', 'b', 'a', 'a'],
    ...         'D': [11.0, 22.0, 33.0, 44.0, 55.0, 66.2],
    ...         'F': [1, 2, 3, 1, 2, 4]})
    >>> X_expected = X[['A', 'B', 'C']].copy()
    >>> y = pd.Series([1, 1, 1, 0, 0, 0], name='TARGET')
    >>> obj = InformationValue(k=3, discretizer=Discretizer(n_bins=4))
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
    >>> y = ks.Series([1, 1, 1, 0, 0, 0], name='TARGET')
    >>> obj = InformationValue(k=3, discretizer=Discretizer(n_bins=4))
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
    >>> X_expected = X[['A', 'B', 'C']].copy()
    >>> y = pd.Series([1, 1, 1, 0, 0, 0], name='TARGET')
    >>> obj = InformationValue(k=3, discretizer=Discretizer(n_bins=4))
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
    >>> y = ks.Series([1, 1, 1, 0, 0, 0], name='TARGET')
    >>> obj = InformationValue(k=3, discretizer=Discretizer(n_bins=4))
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
            raise TypeError("`discretizer` should derive from _BaseDiscretizer.")
        _BaseFeatureSelection.__init__(self)
        self.k = k
        self.discretizer = discretizer

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "InformationValue":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X: Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y: Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
            InformationValue: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.check_binary_target(y)
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
        X: Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y: np.ndarray
            Target values.
        discretizer: _BaseDiscretizer
            Discretizer.

        Returns
        -------
        pd.Series
            Information value.
        """
        discretizer.inplace = False
        object_columns = util.get_datatype_columns(X, object)
        numerical_columns = util.get_numerical_columns(X)
        binned_columns = [f"{c}__bin" for c in numerical_columns]
        iv_columns = object_columns.copy() + binned_columns.copy()
        X = discretizer.fit_transform(X)
        iv = pd.Series(0, index=iv_columns)
        X = X.join(y)
        y_name = y.name
        for col in iv_columns:
            if isinstance(X, pd.DataFrame):
                tab = X.groupby([col, y_name])[y_name].count().unstack().fillna(0)
            else:
                tab = (
                    X[[col, y_name]]
                    .groupby([col, y_name])[y_name]
                    .count()
                    .to_pandas()
                    .unstack()
                    .fillna(0)
                )
            tab /= tab.sum()
            tab = tab.to_numpy()
            with np.errstate(divide="ignore"):
                woe = pd.Series(np.log(tab[:, 1] / tab[:, 0]))
            woe[(woe == np.inf) | (woe == -np.inf)] = 0.0
            iv.loc[col] = np.sum(woe * (tab[:, 1] - tab[:, 0]))

        X = X.drop(binned_columns + [y_name], axis=1)
        iv.index = [c.split("__bin")[0] for c in iv.index]
        return iv.sort_values(ascending=False).fillna(0)
