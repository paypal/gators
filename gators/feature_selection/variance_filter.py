# License: Apache-2.0


from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

from gators import DataFrame, Series


class VarianceFilter(_BaseFeatureSelection):
    """Remove low variance columns.

    Parameters
    ----------
    min_var : float
        Variance threshold.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_selection import VarianceFilter
    >>> obj = VarianceFilter(min_var=0.9)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         B
    0  1.0
    1  2.0
    2  3.0

    >>> X = pd.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
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

    def fit(self, X: DataFrame, y: Series = None) -> "VarianceFilter":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.
        Returns
        -------
        self : "VarianceFilter"
            Instance of itself.
        """
        self.check_dataframe(X)
        numerical_columns = util.get_numerical_columns(X)
        self.feature_importances_ = util.get_function(X).to_pandas(
            X[numerical_columns].var()
        )
        mask = self.feature_importances_ < self.min_var
        self.columns_to_drop = list(self.feature_importances_.index[mask])
        self.selected_columns = util.exclude_columns(X.columns, self.columns_to_drop)
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self
