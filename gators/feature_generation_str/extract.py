# License: Apache-2.0
from typing import List

import numpy as np

from feature_gen_str import extract_str

from ..util import util
from ._base_string_feature import _BaseStringFeature

from gators import DataFrame, Series


class Extract(_BaseStringFeature):
    """Create new object columns based on substrings.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    i_min_vec : List[int]
        List of indices.
    i_max_vec : List[int]
        List of indices.
    column_names : List[int]
        List of column names.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_str import Extract
    >>> obj = Extract(columns=['A', 'A'], i_min_vec=[0, 2], i_max_vec=[1, 3])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A  B A__substring_0_to_1 A__substring_2_to_3
    0  qwe  1                   q                   e
    1  asd  2                   a                   d
    2  zxc  3                   z                   c

    >>> X = pd.DataFrame({'A': ['qwe', 'asd', 'zxc'], 'B': [1, 2, 3]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qwe', 1, 'q', 'e'],
           ['asd', 2, 'a', 'd'],
           ['zxc', 3, 'z', 'c']], dtype=object)
    """

    def __init__(
        self,
        columns: List[str],
        i_min_vec: List[int],
        i_max_vec: List[int],
        column_names: List[int] = None,
    ):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not isinstance(i_min_vec, (list, np.ndarray)):
            raise TypeError("`i_min_vec` should be a list.")
        if len(columns) != len(i_min_vec):
            raise ValueError("Length of `columns` and `i_min_vec` should match.")
        if not isinstance(i_max_vec, (list, np.ndarray)):
            raise TypeError("`i_max_vec` should be a list.")
        if len(columns) != len(i_max_vec):
            raise ValueError("Length of `columns` and `i_max_vec` should match.")
        if not column_names:
            column_names = [
                f"{c}__substring_{i_min}_to_{i_max}"
                for c, i_min, i_max in zip(columns, i_min_vec, i_max_vec)
            ]
        _BaseStringFeature.__init__(self, columns, column_names)
        self.i_min_vec = np.array(i_min_vec, int)
        self.i_max_vec = np.array(i_max_vec, int)

    def fit(self, X: DataFrame, y: Series = None) -> "Extract":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        Extract
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
        )
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

        for col, i_min, i_max, name in zip(
            self.columns, self.i_min_vec, self.i_max_vec, self.column_names
        ):
            X[name] = X[col].str.slice(
                start=i_min, stop=i_max
            )  # .replace({"": "MISSING"})
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
        return extract_str(X, self.idx_columns, self.i_min_vec, self.i_max_vec)
