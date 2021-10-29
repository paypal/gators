# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from scaler import standard_scaler

from ..transformers.transformer import Transformer


class StandardScaler(Transformer):
    """Scale each column by setting the mean to 0 and the standard deviation to 1.



    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    --------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.scalers import StandardScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> obj.fit_transform(X)
         A         B
    0 -1.0 -1.120897
    1  0.0  0.320256
    2  1.0  0.800641

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import StandardScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> obj.fit_transform(X)
         A         B
    0 -1.0 -1.120897
    1  0.0  0.320256
    2  1.0  0.800641

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.scalers import StandardScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.        , -1.12089708],
           [ 0.        ,  0.32025631],
           [ 1.        ,  0.80064077]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import StandardScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = StandardScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.        , -1.12089708],
           [ 0.        ,  0.32025631],
           [ 1.        ,  0.80064077]])

    """

    def __init__(self, dtype: type = np.float64):
        self.dtype = dtype
        self.X_mean: Union[pd.DataFrame, ks.DataFrame] = None
        self.X_std: Union[pd.DataFrame, ks.DataFrame] = None
        self.X_mean_np = np.array([])
        self.X_std_np = np.array([])

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "StandardScaler":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
            'StandardScaler': Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.X_std = X.std().astype(self.dtype)
        self.X_mean = X.mean().astype(self.dtype)
        self.X_mean_np = self.X_mean.to_numpy().astype(self.dtype)
        self.X_std_np = self.X_std.to_numpy().astype(self.dtype)
        return self

    def transform(self, X):
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Transformed dataframe.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)

        def f(x: ks.Series[self.dtype]):
            c = x.name
            return (x - self.X_mean[c]) / self.X_std[c]

        return X.astype(self.dtype).apply(f)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the numpy ndarray X.

        Parameters
        ----------
        X (np.ndarray): Input ndarray.

        Returns
        -------
            np.ndarray: Imputed ndarray.
        """
        self.check_array(X)
        self.check_array_is_numerics(X)
        return standard_scaler(X.astype(self.dtype), self.X_mean_np, self.X_std_np)
