# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from scaler import minmax_scaler

from ..transformers.transformer import Transformer


class MinMaxScaler(Transformer):
    """Scale each column to the [0, 1] range.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.scalers import MinMaxScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> obj.fit_transform(X)
         A     B
    0  0.0  0.00
    1  0.5  0.75
    2  1.0  1.00

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import MinMaxScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> obj.fit_transform(X)
         A     B
    0  0.0  0.00
    1  0.5  0.75
    2  1.0  1.00

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.scalers import MinMaxScaler
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.  , 0.  ],
           [0.5 , 0.75],
           [1.  , 1.  ]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.scalers import MinMaxScaler
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [-0.1, 0.2, 0.3]})
    >>> obj = MinMaxScaler()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.  , 0.  ],
           [0.5 , 0.75],
           [1.  , 1.  ]])

    """

    def __init__(self, dtype: type = np.float64):
        self.dtype = dtype
        self.X_min: Union[pd.DataFrame, ks.DataFrame] = None
        self.X_max: Union[pd.DataFrame, ks.DataFrame] = None
        self.X_min_np = np.array([])
        self.X_max_np = np.array([])

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "MinMaxScaler":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
            'MinMaxScaler': Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.X_min = X.min().astype(self.dtype)
        self.X_max = X.max().astype(self.dtype)
        self.X_min_np = self.X_min.to_numpy().astype(self.dtype)
        self.X_max_np = self.X_max.to_numpy().astype(self.dtype)
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
            return (x - self.X_min.loc[c]) / (self.X_max[c] - self.X_min[c])

        return X.astype(self.dtype).apply(f)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array X.

        Parameters
        ----------
        X (np.ndarray): Input ndarray.

        Returns
        -------
            np.ndarray: Imputed ndarray.
        """
        self.check_array(X)
        return minmax_scaler(X.astype(self.dtype), self.X_min_np, self.X_max_np)
