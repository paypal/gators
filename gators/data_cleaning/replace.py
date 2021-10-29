# License: Apache-2.0
from typing import Dict, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from data_cleaning import replace

from ..transformers.transformer import Transformer
from ..util import util


class Replace(Transformer):
    """Replace the categorical values by the ones given by the user.

    The transformer only accepts categorical columns.

    Parameters
    ----------
    to_replace_dict: Dict[str, Dict[str, str]]
        The dictionary keys are the columns and the dictionary values
        are the `to_replace` dictionary.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.data_cleaning import Replace
    >>> X = pd.DataFrame(
    ...     {'A': ['a', 'b', 'c'], 'B': ['d', 'e','f'],'C': [1, 2, 3]})
    >>> to_replace_dict = {'A': {'a': 'X', 'b': 'Z'}, 'B': {'d': 'Y'}}
    >>> obj = Replace(to_replace_dict=to_replace_dict)
    >>> obj.fit_transform(X)
       A  B  C
    0  X  Y  1
    1  Z  e  2
    2  c  f  3

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import Replace
    >>> X = ks.DataFrame(
    ...     {'A': ['a', 'b', 'c'], 'B': ['d', 'e','f'],'C': [1, 2, 3]})
    >>> to_replace_dict = {'A': {'a': 'X', 'b': 'Z'}, 'B': {'d': 'Y'}}
    >>> obj = Replace(to_replace_dict=to_replace_dict)
    >>> obj.fit_transform(X)
       A  B  C
    0  X  Y  1
    1  Z  e  2
    2  c  f  3

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.data_cleaning import Replace
    >>> X = pd.DataFrame(
    ...     {'A': ['a', 'b', 'c'], 'B': ['d', 'e','f'],'C': [1, 2, 3]})
    >>> to_replace_dict = {'A': {'a': 'X', 'b': 'Z'}, 'B': {'d': 'Y'}}
    >>> obj = Replace(to_replace_dict=to_replace_dict)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['X', 'Y', 1],
           ['Z', 'e', 2],
           ['c', 'f', 3]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import Replace
    >>> X = ks.DataFrame(
    ...     {'A': ['a', 'b', 'c'], 'B': ['d', 'e','f'],'C': [1, 2, 3]})
    >>> to_replace_dict = {'A': {'a': 'X', 'b': 'Z'}, 'B': {'d': 'Y'}}
    >>> obj = Replace(to_replace_dict=to_replace_dict)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['X', 'Y', 1],
           ['Z', 'e', 2],
           ['c', 'f', 3]], dtype=object)


    """

    def __init__(self, to_replace_dict: Dict[str, Dict[str, str]]):
        if not isinstance(to_replace_dict, dict):
            raise TypeError("`to_replace_dict` should be a dict.")
        Transformer.__init__(self)
        self.to_replace_dict = to_replace_dict
        self.columns = list(to_replace_dict.keys())
        n_cols = len(self.columns)
        n_rows = max([len(v) for v in self.to_replace_dict.values()])
        self.to_replace_np_keys = np.empty((n_cols, n_rows), object)
        self.to_replace_np_vals = np.empty((n_cols, n_rows), object)
        self.n_elements_vec = np.empty(n_cols, int)
        for i, col in enumerate(self.to_replace_dict):
            n_elements = len(self.to_replace_dict[col])
            self.n_elements_vec[i] = n_elements
            self.to_replace_np_keys[i, :n_elements] = list(
                self.to_replace_dict[col].keys()
            )[:n_elements]
            self.to_replace_np_vals[i, :n_elements] = list(
                self.to_replace_dict[col].values()
            )[:n_elements]

    def fit(self, X: Union[pd.DataFrame, ks.DataFrame], y=None) -> "Replace":
        """Fit the transformer on the dataframe X.

        Get the list of column names to remove and the array of
          indices to be kept.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        Replace: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_nans(X, self.columns)
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        return self

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame]
    ) -> Union[pd.DataFrame, ks.DataFrame]:
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
        return X.replace(self.to_replace_dict)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.check_array(X)
        return replace(
            X,
            self.idx_columns,
            self.to_replace_np_keys,
            self.to_replace_np_vals,
            self.n_elements_vec,
        )
