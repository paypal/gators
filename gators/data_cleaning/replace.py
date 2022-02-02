# License: Apache-2.0
from typing import Dict

import numpy as np

from data_cleaning import replace

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class Replace(Transformer):
    """Replace the categorical values by the ones given by the user.

    The transformer only accepts categorical columns.

    Parameters
    ----------
    to_replace_dict : Dict[str, Dict[str, str]]
        The dictionary keys are the columns and the dictionary values
        are the `to_replace` dictionary.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.data_cleaning import Replace
    >>> to_replace_dict = {'A': {'a': 'X', 'b': 'Z'}, 'B': {'d': 'Y'}}
    >>> obj = Replace(to_replace_dict=to_replace_dict)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],'C': [1, 2, 3]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame(
    ... {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],'C': [1, 2, 3]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ... {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],'C': [1, 2, 3]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A  B  C
    0  X  Y  1
    1  Z  e  2
    2  c  f  3

    >>> X = pd.DataFrame(
    ... {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],'C': [1, 2, 3]})
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

    def fit(self, X: DataFrame, y: Series = None) -> "Replace":
        """Fit the transformer on the dataframe X.

        Get the list of column names to remove and the array of
          indices to be kept.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : Replace
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
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
        return util.get_function(X).replace(X, self.to_replace_dict)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
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
