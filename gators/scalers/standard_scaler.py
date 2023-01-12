# License: Apache-2.0


from ._base_scaler import _BaseScaler
from ..util import util

from gators import DataFrame, Series


class StandardScaler(_BaseScaler):
    """Scale each column by setting the mean to 0 and the standard deviation to 1.



    Parameters
    ----------
    inplace : bool, default True.
        If True, perform the scaling in-place.
        If False, create new columns.

    Examples
    --------
    Imports and initialization:

    >>> from gators.scalers import StandardScaler
    >>> obj = StandardScaler()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A         B
    0 -1.0 -1.120897
    1  0.0  0.320256
    2  1.0  0.800641

    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.        , -1.12089708],
           [ 0.        ,  0.32025631],
           [ 1.        ,  0.80064077]])
    """

    def __init__(self, inplace: bool = True):
        _BaseScaler.__init__(self, inplace=inplace)

    def fit(self, X: DataFrame, y: Series = None) -> "StandardScaler":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'StandardScaler'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.columns = util.get_numerical_columns(X)
        self.idx_columns = util.get_idx_columns(X, self.columns)
        self.column_names = self.get_column_names(
            self.inplace, self.columns, "standard_scaler"
        )
        self.X_offset = (
            util.get_function(X).to_pandas(X[self.columns].mean()).astype(float)
        )
        self.X_scale = (
            1.0 / util.get_function(X).to_pandas(X[self.columns].std())
        ).astype(float)
        self.X_offset_np = util.get_function(self.X_offset).to_numpy(self.X_offset)
        self.X_scale_np = util.get_function(self.X_scale).to_numpy(self.X_scale)
        self.X_offset = self.X_offset.to_dict()
        self.X_scale = self.X_scale.to_dict()
        return self
