# License: Apache-2.0
from typing import Dict, List

import numpy as np

from clipping import _clipping

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class QuantileClipping(Transformer):
    """Trim values using the quantile values provided by the user.

    The column values will be contained between [*min_quantile*, *max_quantile*],

    Parameters
    ----------
    min_quantile : float
        Lower bound quantile.

    max_quantile : float
        Upper bound quantile.
    Examples
    ---------
    Imports and initialization:

    >>> from gators.clipping import QuantileClipping
    >>> obj = QuantileClipping(columns=["A", "B", "C"], min_quantile=0.2, max_quantile=0.8)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B    C
    0  0.5  0.4  1.0
    1  0.5  0.5 -0.0
    2  0.5 -0.2 -0.0
    3  0.4  0.1  1.0
    4  0.5  0.1  0.4

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.5,  0.4,  1. ],
           [ 0.5,  0.5, -0. ],
           [ 0.5, -0.2, -0. ],
           [ 0.4,  0.1,  1. ],
           [ 0.5,  0.1,  0.4]])
    """

    def __init__(
        self,
        columns: List[str],
        min_quantile: float,
        max_quantile: float,
        inplace: bool = True,
    ):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not isinstance(min_quantile, float):
            raise TypeError("`min_quantile` should be a float between [0, 1].")
        if not isinstance(max_quantile, float):
            raise TypeError("`max_quantile` should be a float between [0, 1].")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        if len(columns) == 0:
            raise ValueError("`columns` should be a not empty list.")
        self.columns = columns
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.inplace = inplace
        self.column_names = self.get_column_names(
            self.inplace, self.columns, "quantile_clip"
        )

    def fit(self, X: DataFrame, y: Series = None) -> "QuantileClipping":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'QuantileClipping'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.idx_columns = util.get_idx_columns(X, self.columns)

        self.clip_dict = {
            c: util.get_function(X).to_numpy(
                X[c].quantile(q=[self.min_quantile, self.max_quantile])
            )
            for c in self.columns
        }
        self.clip_np = np.array(list(self.clip_dict.values()))

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
        for col, name in zip(self.columns, self.column_names):
            X[name] = X[col].clip(self.clip_dict[col][0], self.clip_dict[col][1])

        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

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
        if self.inplace:
            X[:, self.idx_columns] = _clipping(
                X[:, self.idx_columns].astype(float), self.clip_np
            )
            return X
        else:
            X_clip = _clipping(X[:, self.idx_columns].astype(float), self.clip_np)
            return np.concatenate((X, X_clip), axis=1)
