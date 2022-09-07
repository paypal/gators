# License: Apache-2.0
import numpy as np

from typing import Dict

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class RenameColumns(Transformer):
    """Rename the columns.

    Parameters
    ----------
    to_rename_dict : Dict[str, Dict[str, str]]
        The dictionary keys are the columns and its values
        are the new columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.data_cleaning import RenameColumns
    >>> to_rename_dict = {'A': 'A_f', 'B': 'B_f'}
    >>> obj = RenameColumns(to_rename_dict=to_rename_dict)

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
      A_f B_f  C
    0   a   d  1
    1   b   e  2
    2   c   f  3

    >>> X = pd.DataFrame(
    ... {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],'C': [1, 2, 3]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'd', 1],
           ['b', 'e', 2],
           ['c', 'f', 3]], dtype=object)
    """

    def __init__(self, to_rename_dict: Dict[str, Dict[str, str]]):
        if not isinstance(to_rename_dict, dict):
            raise TypeError("`to_rename_dict` should be a dict.")
        Transformer.__init__(self)
        self.to_rename_dict = to_rename_dict
        self.columns = list(to_rename_dict.keys())

    def fit(self, X: DataFrame, y: Series = None) -> "Rename":
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
        self : Rename
            Instance of itself.
        """
        self.check_dataframe(X)
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
        return X.rename(columns=self.to_rename_dict)

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
        return X
