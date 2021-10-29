# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import is_null, is_null_object

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration


class IsNull(_BaseFeatureGeneration):
    """Create new columns based on missing values.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    dtype : type, default to np.float64
        Numpy dtype of the output columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation import IsNull
    >>> X = pd.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})
    >>> obj = IsNull(columns=['A', 'B'])
    >>> obj.fit_transform(X)
          A    B  A__is_null  B__is_null
    0  None  NaN         1.0         1.0
    1     a  1.0         0.0         0.0
    2     b  1.0         0.0         0.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import IsNull
    >>> X = ks.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})
    >>> obj = IsNull(columns=['A', 'B'])
    >>> obj.fit_transform(X)
          A    B  A__is_null  B__is_null
    0  None  NaN         1.0         1.0
    1     a  1.0         0.0         0.0
    2     b  1.0         0.0         0.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import IsNull
    >>> X = pd.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})
    >>> obj = IsNull(columns=['A', 'B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[None, nan, 1.0, 1.0],
           ['a', 1.0, 0.0, 0.0],
           ['b', 1.0, 0.0, 0.0]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import IsNull
    >>> X = ks.DataFrame({'A': [None, 'a', 'b'], 'B': [np.nan, 1, 1]})
    >>> obj = IsNull(columns=['A', 'B'])
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
        dtype: type = np.float64,
    ):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if column_names is not None and not isinstance(column_names, list):
            raise TypeError("`column_names` should be a list.")
        if not column_names:
            column_names = [f"{c}__is_null" for c in columns]
        if len(column_names) != len(columns):
            raise ValueError("Length of `columns` and `column_names` should match.")
        column_mapping = dict(zip(column_names, columns))
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
            column_mapping=column_mapping,
            dtype=dtype,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ):
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
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
        if isinstance(X, pd.DataFrame):
            X[self.column_names] = X[self.columns].isnull().astype(self.dtype)
            return X
        for col, name in zip(self.columns, self.column_names):
            X = X.assign(dummy=X[col].isnull().astype(self.dtype)).rename(
                columns={"dummy": name}
            )
        return X

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
        if X.dtype == object:
            return is_null_object(X, self.idx_columns)
        return is_null(X, self.idx_columns, self.dtype)
