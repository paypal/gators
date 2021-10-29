# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import pandas as pd

from ..util import util
from ._base_feature_selection import _BaseFeatureSelection


class VarianceFilter(_BaseFeatureSelection):
    """Remove low variance columns.

    Parameters
    ----------
    min_var : float
        Variance threshold.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_selection import VarianceFilter
    >>> X = pd.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = VarianceFilter(min_var=0.9)
    >>> obj.fit_transform(X)
         B
    0  1.0
    1  2.0
    2  3.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_selection import VarianceFilter
    >>> X = pd.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = VarianceFilter(min_var=0.9)
    >>> obj.fit_transform(X)
         B
    0  1.0
    1  2.0
    2  3.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_selection import VarianceFilter
    >>> X = pd.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = VarianceFilter(min_var=0.9)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.],
           [2.],
           [3.]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_selection import VarianceFilter
    >>> X = ks.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = VarianceFilter(min_var=0.9)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.],
           [2.],
           [3.]])

    """

    def __init__(self, min_var: float):
        if not isinstance(min_var, float):
            raise TypeError("`min_var` should be a float.")
        _BaseFeatureSelection.__init__(self)
        self.min_var = min_var

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "VarianceFilter":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
            VarianceFilter: Instance of itself.
        """
        self.check_dataframe(X)
        numerical_columns = util.get_numerical_columns(X)
        self.feature_importances_ = X[numerical_columns].var()
        if isinstance(self.feature_importances_, ks.Series):
            self.feature_importances_ = self.feature_importances_.to_pandas()
        mask = self.feature_importances_ < self.min_var
        self.columns_to_drop = list(self.feature_importances_.index[mask])
        self.selected_columns = util.exclude_columns(X.columns, self.columns_to_drop)
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self
