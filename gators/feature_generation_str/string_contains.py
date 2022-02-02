# License: Apache-2.0
from typing import List

import numpy as np
import pandas as pd

from feature_gen_str import contains

from ..util import util
from ._base_string_feature import _BaseStringFeature

from gators import DataFrame, Series


class StringContains(_BaseStringFeature):
    """Create new binary columns.

    The value is 1 if the element contains the given substring and 0 otherwise.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    contains_vec : List[int]
        List of substrings.
    column_names : List[int], default None.
        List new column names.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_str import StringContains
    >>> obj = StringContains(columns=['A', 'A'], contains_vec=['qw', 'we'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(
    ... pd.DataFrame({'A': ['qwe', 'qwd', 'zwe'], 'B': [1, 2, 3]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['qwe', 'qwd', 'zwe'], 'B': [1, 2, 3]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['qwe', 'qwd', 'zwe'], 'B': [1, 2, 3]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A  B  A__contains_qw  A__contains_we
    0  qwe  1             1.0             1.0
    1  qwd  2             1.0             0.0
    2  zwe  3             0.0             1.0

    >>> X = pd.DataFrame({'A': ['qwe', 'qwd', 'zwe'], 'B': [1, 2, 3]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qwe', 1, 1.0, 1.0],
           ['qwd', 2, 1.0, 0.0],
           ['zwe', 3, 0.0, 1.0]], dtype=object)
    """

    def __init__(
        self,
        columns: List[str],
        contains_vec: List[str],
        column_names: List[str] = None,
    ):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not isinstance(contains_vec, (list, np.ndarray)):
            raise TypeError("`contains_vec` should be a list.")
        if len(columns) != len(contains_vec):
            raise ValueError("Length of `columns` and `contains_vec` should match.")
        if not column_names:
            column_names = [
                f"{col}__contains_{val}" for col, val in zip(columns, contains_vec)
            ]
        _BaseStringFeature.__init__(self, columns, column_names)
        self.contains_vec = contains_vec
        self.contains_vec_np = np.array(self.contains_vec).astype(object)

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
        for col, val, name in zip(self.columns, self.contains_vec, self.column_names):
            X[name] = X[col].str.contains(val, regex=False).astype(np.float64)
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
        return contains(X, self.idx_columns, self.contains_vec_np)
