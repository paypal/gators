from typing import Tuple

import numpy as np

from ..transformers.transformer_xy import TransformerXY
from ..util import util

from gators import DataFrame, Series


class ToPandas(TransformerXY):
    """Convert dataframe and series to a pandas dataframe and series.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.converter import ToPandas
    >>> obj = ToPandas()

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

    The result is a Pandas dataframe and Pandas series.

    >>> X, y = obj.transform(X, y)
    >>> X
         A    B    C
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
        X: DataFrame,
        y: Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
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
        self.check_dataframe(X)
        self.check_target(X, y)
        return util.get_function(X).to_pandas(X), util.get_function(X).to_pandas(y)
