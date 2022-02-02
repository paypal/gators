# License: Apache-2.0
from typing import List

import numpy as np

from feature_gen import is_equal, is_equal_object

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class IsEqual(_BaseFeatureGeneration):
    """Create new columns based on value matching.

    Parameters
    ----------
    columns_a : List[str]
        List of columns.
    columns_b : List[str]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import IsEqual
    >>> obj = IsEqual(columns_a=['A'],columns_b=['B'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A  B  A__is__B
    0  1  1       1.0
    1  2  1       0.0
    2  3  1       0.0

    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1, 1, 1],
           [2, 1, 0],
           [3, 1, 0]])
    """

    def __init__(
        self, columns_a: List[str], columns_b: List[str], column_names: List[str] = None
    ):
        if not isinstance(columns_a, (list, np.ndarray)):
            raise TypeError("`columns_a` should be a list.")
        if not isinstance(columns_b, (list, np.ndarray)):
            raise TypeError("`columns_b` should be a list.")
        if column_names is not None and not isinstance(column_names, list):
            raise TypeError("`columns_a` should be a list.")
        if len(columns_a) != len(columns_b):
            raise ValueError("Length of `columns_a` and `columns_b` should match.")
        if len(columns_a) == 0:
            raise ValueError("`columns_a` and `columns_b` should not be empty.")
        if not column_names:
            column_names = [
                f"{c_a}__is__{c_b}" for c_a, c_b in zip(columns_a, columns_b)
            ]
        if len(columns_a) != len(column_names):
            raise ValueError(
                """Length of `columns_a`, `columns_b` and `column_names`
                should match."""
            )
        columns = list(set(columns_a + columns_b))
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.idx_columns_a: List[int] = []
        self.idx_columns_b: List[int] = []

    def fit(self, X: DataFrame, y: Series = None):
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        IsEqual
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns_a = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns_a
        )
        self.idx_columns_b = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns_b
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
        for a, b, name in zip(self.columns_a, self.columns_b, self.column_names):
            X[name] = (X[a] == X[b]).astype(float)
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
        if X.dtype == object:
            return is_equal_object(X, self.idx_columns_a, self.idx_columns_b)
        return is_equal(X, self.idx_columns_a, self.idx_columns_b)
