from typing import Tuple

import numpy as np

from ..transformers.transformer_xy import TransformerXY
from ..util import util

from gators import DataFrame, Series


class ToNumpy(TransformerXY):
    """Convert dataframe and series to NumPy arrays.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.converter import ToNumpy
    >>> obj = ToNumpy()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [0.0, 3.0, 6.0],
    ... 'B': [1.0, 4.0, 7.0],
    ... 'C': [2.0, 5.0, 8.0]}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([0, 0, 1], name='TARGET'), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': [0.0, 3.0, 6.0],
    ... 'B': [1.0, 4.0, 7.0],
    ... 'C': [2.0, 5.0, 8.0]})
    >>> y = ks.Series([0, 0, 1], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': [0.0, 3.0, 6.0],
    ... 'B': [1.0, 4.0, 7.0],
    ... 'C': [2.0, 5.0, 8.0]})
    >>> y = pd.Series([0, 0, 1], name='TARGET')

    The result is a 2D NumPy array for `X` and a 1D NumPy array for `y`.

    >>> X, y = obj.transform(X, y)
    >>> X
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> y
    array([0, 0, 1])
    """

    def __init__(self):
        TransformerXY.__init__(self)

    def transform(
        self,
        X: DataFrame,
        y: Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Dataframe.
        y : [pd.Series, ks.Series]:
            Target values.

        Returns
        -------
        X : np.ndarray
             Array.
        y : np.ndarray
            Target values.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        return util.get_function(X).to_numpy(X), util.get_function(X).to_numpy(y)
