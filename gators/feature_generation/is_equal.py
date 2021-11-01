# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import is_equal, is_equal_object

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration


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
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation import IsEqual
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = IsEqual(columns_a=['A'],columns_b=['B'])
    >>> obj.fit_transform(X)
       A  B  A__is__B
    0  1  1         1
    1  2  1         0
    2  3  1         0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import IsEqual
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = IsEqual(columns_a=['A'],columns_b=['B'])
    >>> obj.fit_transform(X)
       A  B  A__is__B
    0  1  1         1
    1  2  1         0
    2  3  1         0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import IsEqual
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = IsEqual(columns_a=['A'],columns_b=['B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1, 1, 1],
           [2, 1, 0],
           [3, 1, 0]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import IsEqual
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = IsEqual(columns_a=['A'],columns_b=['B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1, 1, 1],
           [2, 1, 0],
           [3, 1, 0]])

    """

    def __init__(
        self, columns_a: List[str], columns_b: List[str], column_names: List[str] = None
    ):
        if not isinstance(columns_a, list):
            raise TypeError("`columns_a` should be a list.")
        if not isinstance(columns_b, list):
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
        column_mapping = {
            name: [c_a, c_b]
            for name, c_a, c_b in zip(column_names, columns_a, columns_b)
        }
        columns = list(set(columns_a + columns_b))
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
            column_mapping=column_mapping,
            dtype=None,
        )
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.idx_columns_a: List[int] = []
        self.idx_columns_b: List[int] = []

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ):
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

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
            for a, b, name in zip(self.columns_a, self.columns_b, self.column_names):
                x_dtype = X[a].dtype
                x_dtype = (
                    x_dtype if (x_dtype != object) and (x_dtype != bool) else np.float64
                )
                X.loc[:, name] = (X[a] == X[b]).astype(x_dtype)
            return X

        for a, b, name in zip(self.columns_a, self.columns_b, self.column_names):
            x_dtype = X[a].dtype
            x_dtype = (
                x_dtype if (x_dtype != object) and (x_dtype != bool) else np.float64
            )
            X = X.assign(dummy=(X[a] == X[b]).astype(x_dtype)).rename(
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
            return is_equal_object(X, self.idx_columns_a, self.idx_columns_b)
        return is_equal(X, self.idx_columns_a, self.idx_columns_b)
