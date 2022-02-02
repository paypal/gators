# License: Apache-2.0


import numpy as np

from scaler import standard_scaler

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class StandardScaler(Transformer):
    """Scale each column by setting the mean to 0 and the standard deviation to 1.



    Parameters
    ----------
    dtype : type, default np.float64.
        Numerical datatype of the output data.

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

    def __init__(self, dtype: type = np.float64):
        self.dtype = dtype
        self.X_mean: DataFrame = None
        self.X_std: DataFrame = None
        self.X_mean_np = np.array([])
        self.X_std_np = np.array([])

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
        self.check_dataframe_is_numerics(X)
        self.columns = list(X.columns)
        self.X_mean = util.get_function(X).to_pandas(X.mean()).astype(self.dtype)
        self.X_std = util.get_function(X).to_pandas(X.std()).astype(self.dtype)
        self.X_mean_np = util.get_function(self.X_mean).to_numpy(self.X_mean)
        self.X_std_np = util.get_function(self.X_std).to_numpy(self.X_std)
        self.X_mean = self.X_mean.to_dict()
        self.X_std = self.X_std.to_dict()
        return self

    def transform(self, X):
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)

        for col in self.columns:
            X[col] = (X[col] - self.X_mean[col]) / self.X_std[col]
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array X.

        Parameters
        ----------
        X :np.ndarray:
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        self.check_array_is_numerics(X)
        return standard_scaler(X.astype(self.dtype), self.X_mean_np, self.X_std_np)
