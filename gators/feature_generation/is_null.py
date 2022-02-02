# License: Apache-2.0
from typing import List

import numpy as np

from feature_gen import is_null, is_null_object

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class IsNull(_BaseFeatureGeneration):
    """Create new columns based on missing values.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    dtype : type, default np.float64
        Numpy dtype of the output columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import IsNull
    >>> obj = IsNull(columns=['A', 'B'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
          A    B  A__is_null  B__is_null
    0  None  NaN         1.0         1.0
    1     a  1.0         0.0         0.0
    2     b  1.0         0.0         0.0


    >>> X = pd.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[None, nan, 1.0, 1.0],
           ['a', 1.0, 0.0, 0.0],
           ['b', 1.0, 0.0, 0.0]], dtype=object)
    """

    def __init__(
        self,
        columns: List[str],
        column_names: List[str] = None,
    ):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if column_names is not None and not isinstance(
            column_names, (list, np.ndarray)
        ):
            raise TypeError("`column_names` should be a list.")
        if not column_names:
            column_names = [f"{c}__is_null" for c in columns]
        if len(column_names) != len(columns):
            raise ValueError("Length of `columns` and `column_names` should match.")
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )

    def fit(self, X: DataFrame, y: Series = None):
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
            y (np.ndarray, optional): labels. Defaults to None.

        Returns
        -------
        IsNull:
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
        for col, name in zip(self.columns, self.column_names):
            X[name] = X[col].isnull().astype(np.float64)
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
            return is_null_object(X, self.idx_columns)
        return is_null(X, self.idx_columns)
