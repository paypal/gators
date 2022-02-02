# License: Apache-2.0
from typing import Dict, List

import numpy as np

from clipping import clipping

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class Clipping(Transformer):
    """Trim values using the limits given by the user.

    The data should be only composed of numerical columns.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `Clipping`.

    Parameters
    ----------
    clip_dict : Dict[str, List[float]]
        The keys are the columns to clip, the values are lists of two elements:

        * the first element is the lower limit
        * the second element is the upper limit

    dtype : type, default np.float64.
        Numerical datatype of the output data.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.binning import BinRareCategories
    >>> clip_dict = {'A':[-0.5, 0.5], 'B':[-0.5, 0.5], 'C':[-0., 1.]}
    >>> obj = Clipping(clip_dict=clip_dict)

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

    def __init__(self, clip_dict: Dict[str, List[float]], dtype: type = np.float64):
        if not isinstance(clip_dict, dict):
            raise TypeError("`clip_dict` should be a dictionary.")
        if len(clip_dict) == 0:
            raise ValueError("Length of `clip_dict` should be not zero.")
        self.clip_dict = clip_dict
        self.dtype = dtype
        self.clip_np = np.array(list(clip_dict.values())).astype(self.dtype)
        self.columns = list(clip_dict.keys())

    def fit(self, X: DataFrame, y: Series = None) -> "Clipping":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'Clipping'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.idx_columns = util.get_idx_columns(X, self.clip_dict.keys())
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
        self.columns_ = list(X.columns)

        for col in self.columns:
            X[col] = (
                X[col]
                .clip(self.clip_dict[col][0], self.clip_dict[col][1])
                .astype(self.dtype)
            )
        self.columns_ = list(X.columns)
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
        return clipping(X, self.idx_columns, self.clip_np)
