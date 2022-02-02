# License: Apache-2.0
from typing import List

import numpy as np
import pandas as pd

from feature_gen_str import split_and_extract_str

from ..util import util
from ._base_string_feature import _BaseStringFeature

from gators import DataFrame, Series


class SplitExtract(_BaseStringFeature):
    """Create new columns based on split strings.

    The transformer applies two steps:

    * split each string element using the given value.
    * extract the string of the given split list element.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    str_split_vec : List[int]
        List of separators.

    idx_split_vec : List[int]
        List of split indices.

    column_names : List[int]
        List of new column names.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_str import SplitExtract
    >>> obj = SplitExtract(
    ... columns=['A', 'A'], str_split_vec=['*', '*'], idx_split_vec=[0, 1])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(
    ... pd.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
          A  B A__split_by_*_idx_0 A__split_by_*_idx_1
    0  qw*e  1                  qw                   e
    1  a*qd  2                   a                  qd
    2  zxq*  3                 zxq
    >>> X = pd.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qw*e', 1, 'qw', 'e'],
           ['a*qd', 2, 'a', 'qd'],
           ['zxq*', 3, 'zxq', '']], dtype=object)
    """

    def __init__(
        self,
        columns: List[str],
        str_split_vec: List[int],
        idx_split_vec: List[int],
        column_names: List[str] = None,
    ):

        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not isinstance(str_split_vec, (list, np.ndarray)):
            raise TypeError("`str_split_vec` should be a list.")
        if len(columns) != len(str_split_vec):
            raise ValueError("Length of `columns` and `str_split_vec` should match.")
        if not isinstance(idx_split_vec, (list, np.ndarray)):
            raise TypeError("`idx_split_vec` should be a list.")
        if len(columns) != len(idx_split_vec):
            raise ValueError("Length of `columns` and `idx_split_vec` should match.")
        if not column_names:
            column_names = [
                f"{col}__split_by_{split}_idx_{idx}"
                for col, split, idx in zip(columns, str_split_vec, idx_split_vec)
            ]
        self.str_split_vec = str_split_vec
        self.idx_split_vec = idx_split_vec
        self.str_split_vec_np = np.array(str_split_vec, object)
        self.idx_split_vec_np = np.array(idx_split_vec, int)
        _BaseStringFeature.__init__(self, columns, column_names)

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
        for col, idx, str_split, name in zip(
            self.columns, self.idx_split_vec, self.str_split_vec, self.column_names
        ):
            X[name] = X[col].str.split(str_split).str.get(idx).fillna("MISSING")
        self.columns_ = list(X.columns)
        return X

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
        return split_and_extract_str(
            X, self.idx_columns, self.str_split_vec_np, self.idx_split_vec_np
        )
