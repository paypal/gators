from typing import Tuple, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers.transformer_xy import TransformerXY


class KoalasToPandas(TransformerXY):
    """Convert koalas dataframe and series to a pandas dataframe and series.

    Examples
    ---------
    * transform with pandas

    >>> import databricks.koalas as ks
    >>> from gators.converter import KoalasToPandas
    >>> X = ks.DataFrame({
    ... 'q': {0: 0.0, 1: 3.0, 2: 6.0},
    ... 'w': {0: 1.0, 1: 4.0, 2: 7.0},
    ... 'e': {0: 2.0, 1: 5.0, 2: 8.0}})
    >>> y = ks.Series([0, 0, 1], name='TARGET')
    >>> obj = KoalasToPandas()
    >>> X, y = obj.transform(X, y)
    >>> X
         q    w    e
    0  0.0  1.0  2.0
    1  3.0  4.0  5.0
    2  6.0  7.0  8.0
    >>> y
    0    0
    1    0
    2    1
    Name: TARGET, dtype: int64
    """

    def __init__(self):
        TransformerXY.__init__(self)

    def transform(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : [pd.Series, ks.Series]:
            Target values.

        Returns
        -------
        X : pd.DataFrame
            Dataframe.
        y : np.ndarray
            Target values.
        """
        if not isinstance(X, ks.DataFrame):
            raise TypeError("`X` should be a koalas dataframe")
        self.check_dataframe(X)
        self.check_y(X, y)
        return X.to_pandas(), y.to_pandas()
