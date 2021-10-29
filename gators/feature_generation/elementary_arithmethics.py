# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import elementary_arithmetics
from gators.feature_generation._base_feature_generation import _BaseFeatureGeneration

EPSILON = 1e-10


class ElementaryArithmetics(_BaseFeatureGeneration):
    """Create new columns based on elementary arithmetics.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `ElementaryArithmetics`.

    Parameters
    ----------
    columns_a : List[str]
        List of columns.
    columns_b : List[str]
        List of columns.
    operator : str
        Arithmetic perator. The possible values are:

        * '+' for addition
        * '*' for multiplication
        * '/' for division

    column_names : List[str], default to None.
        List of new column names.
    coef : float, default to 1.
        Coefficient value for the addition.

            X[new] = X[column_a] + coef * X[column_b]

    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------
    * fit & transform with `pandas`

        - addition

            >>> import pandas as pd
            >>> from gators.feature_generation import ElementaryArithmetics
            >>> X = pd.DataFrame({'A': [1, 1., 1.], 'B': [1., 2., 3.]})
            >>> obj = ElementaryArithmetics(
            ... columns_a=['A'], columns_b=['B'], operator='+', coef=0.1)
            >>> obj.fit_transform(X)
                 A    B  A__+__B
            0  1.0  1.0      1.1
            1  1.0  2.0      1.2
            2  1.0  3.0      1.3

        - division

            >>> import pandas as pd
            >>> from gators.feature_generation import ElementaryArithmetics
            >>> X = pd.DataFrame({'A': [1., 1., 1.], 'B': [1., 2., 3.]})
            >>> obj = ElementaryArithmetics(
            ... columns_a=['A'], columns_b=['B'], operator='/')
            >>> obj.fit_transform(X)
                 A    B   A__/__B
            0  1.0  1.0  1.000000
            1  1.0  2.0  0.500000
            2  1.0  3.0  0.333333

        - multiplication & setting new column name

            >>> import pandas as pd
            >>> from gators.feature_generation import ElementaryArithmetics
            >>> X = pd.DataFrame({'A': [1., 2., 3.], 'B': [1., 4., 9.]})
            >>> obj = ElementaryArithmetics(
            ... columns_a=['A'], columns_b=['B'],
            ... operator='*', column_names=['mult'])
            >>> obj.fit_transform(X)
                 A    B  mult
            0  1.0  1.0   1.0
            1  2.0  4.0   8.0
            2  3.0  9.0  27.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import ElementaryArithmetics
    >>> X = ks.DataFrame({'A': [1., 1., 1.], 'B': [1., 2., 3.]})
    >>> obj = ElementaryArithmetics(
    ... columns_a=['A'], columns_b=['B'], operator='/')
    >>> obj.fit_transform(X)
         A    B   A__/__B
    0  1.0  1.0  1.000000
    1  1.0  2.0  0.500000
    2  1.0  3.0  0.333333

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import ElementaryArithmetics
    >>> X = pd.DataFrame({'A': [1., 1., 1.], 'B': [1., 2., 3.]})
    >>> obj = ElementaryArithmetics(
    ... columns_a=['A'], columns_b=['B'], operator='/')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.        , 1.        , 1.        ],
           [1.        , 2.        , 0.5       ],
           [1.        , 3.        , 0.33333333]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import ElementaryArithmetics
    >>> X = ks.DataFrame({'A': [1., 1., 1.], 'B': [1., 2., 3.]})
    >>> obj = ElementaryArithmetics(
    ... columns_a=['A'], columns_b=['B'], operator='/')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.        , 1.        , 1.        ],
           [1.        , 2.        , 0.5       ],
           [1.        , 3.        , 0.33333333]])

    """

    def __init__(
        self,
        columns_a: List[str],
        columns_b: List[str],
        operator: str,
        column_names: List[str] = None,
        coef: float = 1.0,
        dtype: type = np.float64,
    ):
        if not isinstance(columns_a, list):
            raise TypeError("`columns_a` should be a list.")
        if not isinstance(columns_b, list):
            raise TypeError("`columns_b` should be a list.")
        if len(columns_a) == 0:
            raise ValueError("`columns_a` should not be empty.")
        if not isinstance(operator, str):
            raise TypeError("`operator` should be a str.")
        if not isinstance(coef, float):
            raise TypeError("`coef` should be a float.")
        if column_names and not isinstance(column_names, list):
            raise TypeError("`column_names` should be a list.")
        if len(columns_a) != len(columns_b):
            raise ValueError("Length of `columns_a` and `columns_a` should match.")
        if operator not in ["+", "*", "/"]:
            raise ValueError('`operator` should be "+", "*", or "/".')
        if not column_names:
            str_operator = operator
            if coef < 0:
                str_operator = "-"
            column_names = [
                f"{c_a}__{str_operator}__{c_b}"
                for c_a, c_b in zip(columns_a, columns_b)
            ]
            column_mapping = {
                f"{c_a}__{str_operator}__{c_b}": [c_a, c_b]
                for c_a, c_b in zip(columns_a, columns_b)
            }
        else:
            column_mapping = {
                c: [c_a, c_b] for c, c_a, c_b in zip(column_names, columns_a, columns_b)
            }
        if len(column_names) != len(columns_a):
            raise ValueError(
                """Length of `columns_a`, `columns_b`,
                and `column_names` should match."""
            )
        self.check_datatype(dtype, [np.float32, np.float64])
        columns = list(set(columns_a + columns_b))
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
            column_mapping=column_mapping,
            dtype=dtype,
        )
        self.columns = list(set(columns_a + columns_b))
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.idx_columns_a: np.ndarray = np.array([])
        self.idx_columns_b: np.ndarray = np.array([])
        self.operator = operator
        self.coef = coef

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "ElementaryArithmetics":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
        ElementaryArithmetics
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.idx_columns_a = self.get_idx_columns(
            columns=X.columns, selected_columns=self.columns_a
        )
        self.idx_columns_b = self.get_idx_columns(
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
        self.check_dataframe_is_numerics(X)
        for c_a, c_b, c in zip(self.columns_a, self.columns_b, self.column_names):
            if self.operator == "+":
                X[c] = X[c_a] + self.coef * X[c_b]
            elif self.operator == "*":
                X[c] = X[c_a] * X[c_b]
            else:
                X[c] = X[c_a] / (X[c_b] + EPSILON)
            X[c] = X[c]
        X = X.astype(self.dtype)
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
        return elementary_arithmetics(
            X.astype(self.dtype),
            self.idx_columns_a,
            self.idx_columns_b,
            self.operator,
            self.coef,
            EPSILON,
            self.dtype,
        )

    @staticmethod
    def get_idx_columns(columns: List[str], selected_columns: List[str]) -> np.ndarray:
        """Get the indices of the columns used for the combination.

        Parameters
        ----------
        columns : List[str]
            List of columns.
        selected_columns : List[str]
            List of columns.

        Returns:
        np.ndarray
            Array of indices.
        """
        idx = []
        for selected_column in selected_columns:
            for i, column in enumerate(columns):
                if column == selected_column:
                    idx.append(i)
                    break
        return np.array(idx)
