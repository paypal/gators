# License: Apache-2.0
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from clipping import clipping

from ..transformers.transformer import Transformer
from ..util import util


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

    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.clipping import Clipping
    >>> X = pd.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]})
    >>> clip_dict = {'A':[-0.5, 0.5], 'B':[-0.5, 0.5], 'C':[-0., 1.]}
    >>> obj = Clipping(clip_dict=clip_dict)
    >>> obj.fit_transform(X)
         A    B    C
    0  0.5  0.4  1.0
    1  0.5  0.5 -0.0
    2  0.5 -0.2 -0.0
    3  0.4  0.1  1.0
    4  0.5  0.1  0.4

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.clipping import Clipping
    >>> X = ks.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]})
    >>> clip_dict = {'A':[-0.5, 0.5], 'B':[-0.5, 0.5], 'C':[-0., 1.]}
    >>> obj = Clipping(clip_dict=clip_dict)
    >>> obj.fit_transform(X)
         A    B    C
    0  0.5  0.4  1.0
    1  0.5  0.5 -0.0
    2  0.5 -0.2 -0.0
    3  0.4  0.1  1.0
    4  0.5  0.1  0.4

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.clipping import Clipping
    >>> X = pd.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]})
    >>> clip_dict = {'A':[-0.5, 0.5], 'B':[-0.5, 0.5], 'C':[-0., 1.]}
    >>> obj = Clipping(clip_dict=clip_dict)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.5,  0.4,  1. ],
           [ 0.5,  0.5, -0. ],
           [ 0.5, -0.2, -0. ],
           [ 0.4,  0.1,  1. ],
           [ 0.5,  0.1,  0.4]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.clipping import Clipping
    >>> X = ks.DataFrame(
    ... {'A': [1.8, 2.2, 1.0, 0.4, 0.8],
    ... 'B': [0.4, 1.9, -0.2, 0.1, 0.1],
    ... 'C': [1.0, -1.0, -0.1, 1.5, 0.4]})
    >>> clip_dict = {'A':[-0.5, 0.5], 'B':[-0.5, 0.5], 'C':[-0., 1.]}
    >>> obj = Clipping(clip_dict=clip_dict)
    >>> _ = obj.fit(X)
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
        self.clip_np = np.array(list(clip_dict.values()))
        self.columns = list(clip_dict.keys())

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "Clipping":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
            'Clipping': Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.idx_columns = util.get_idx_columns(X, self.clip_dict.keys())
        self.clip_np = self.clip_np.astype(self.dtype)
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

        def clipping(x: ks.Series[self.dtype]) -> ks.Series[self.dtype]:
            c = x.name
            if c not in self.columns:
                return x
            return x.clip(self.clip_dict[c][0], self.clip_dict[c][1])

        return X.apply(clipping)

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
        return clipping(X, self.idx_columns, self.clip_np)
