# License: Apache-2.0
from typing import List

import numpy as np

from feature_gen import elementary_arithmetics
from gators.feature_generation._base_feature_generation import _BaseFeatureGeneration
from gators import DataFrame, Series

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

    column_names : List[str], default None.
        List of new column names.
    coef : float, default 1.
        Coefficient value for the addition.

            X[new] = X[column_a] + coef * X[column_b]

    dtype : type, default np.float64.
        Numerical datatype of the output data.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import ClusterStatistics
    >>> obj = ElementaryArithmetics(
    ... columns_a=['A'], columns_b=['B'], operator='+', coef=0.1)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [1, 1., 1.], 'B': [1., 2., 3.]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [1, 1., 1.], 'B': [1., 2., 3.]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 1., 1.], 'B': [1., 2., 3.]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B  A+0.1xB
    0  1.0  1.0      1.1
    1  1.0  2.0      1.2
    2  1.0  3.0      1.3

    >>> X = pd.DataFrame({'A': [1, 1., 1.], 'B': [1., 2., 3.]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1. , 1. , 1.1],
           [1. , 2. , 1.2],
           [1. , 3. , 1.3]])
    """

    def __init__(
        self,
        columns_a: List[str],
        columns_b: List[str],
        operator: str,
        column_names: List[str] = None,
        coef: float = 1.0,
    ):
        if not isinstance(columns_a, (list, np.ndarray)):
            raise TypeError("`columns_a` should be a list.")
        if not isinstance(columns_b, (list, np.ndarray)):
            raise TypeError("`columns_b` should be a list.")
        if len(columns_a) == 0:
            raise ValueError("`columns_a` should not be empty.")
        if not isinstance(operator, str):
            raise TypeError("`operator` should be a str.")
        if not isinstance(coef, (int, float)):
            raise TypeError("`coef` should be an int or a float.")
        if column_names and not isinstance(column_names, (list, np.ndarray)):
            raise TypeError("`column_names` should be a list.")
        if len(columns_a) != len(columns_b):
            raise ValueError("Length of `columns_a` and `columns_a` should match.")
        if operator not in ["+", "*", "/"]:
            raise ValueError('`operator` should be "+", "*", or "/".')
        if not column_names:
            str_operator = self.get_str_operator(operator, coef)
            column_names = [
                f"{c_a}{str_operator}{c_b}" for c_a, c_b in zip(columns_a, columns_b)
            ]
        if len(column_names) != len(columns_a):
            raise ValueError(
                """Length of `columns_a`, `columns_b`,
                and `column_names` should match."""
            )
        columns = list(set(columns_a + columns_b))
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.idx_columns_a = np.array([])
        self.idx_columns_b = np.array([])
        self.idx_subarray = np.array([])
        self.operator = operator
        self.coef = coef

    def fit(self, X: DataFrame, y: Series = None) -> "ElementaryArithmetics":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        ElementaryArithmetics
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = [
            c for c in X.columns if c in list(set(self.columns_a + self.columns_b))
        ]
        self.check_dataframe_is_numerics(X[self.columns])

        self.idx_subarray = self.get_idx_columns(
            columns=X.columns,
            selected_columns=self.columns,
        )
        self.idx_columns_a = self.get_idx_columns(
            columns=self.columns, selected_columns=self.columns_a
        )
        self.idx_columns_b = self.get_idx_columns(
            columns=self.columns, selected_columns=self.columns_b
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
        for c_a, c_b, c in zip(self.columns_a, self.columns_b, self.column_names):
            if self.operator == "+":
                X[c] = X[c_a] + self.coef * X[c_b]
            elif self.operator == "*":
                X[c] = X[c_a] * X[c_b]
            else:
                X[c] = X[c_a] / (X[c_b] + EPSILON)
            X[c] = X[c]
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
        X_new = elementary_arithmetics(
            X[:, self.idx_subarray].astype(np.float64),
            self.idx_columns_a,
            self.idx_columns_b,
            self.operator,
            self.coef,
            EPSILON,
        )
        return np.concatenate((X, X_new), axis=1)

    @staticmethod
    def get_idx_columns(columns: List[str], selected_columns: List[str]) -> np.ndarray:
        """Get the indices of the columns used for the combination.

        Parameters
        ----------
        theta_vec : List[float]
            List of columns.
        selected_theta_vec : List[float]
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

    @staticmethod
    def get_str_operator(operator, coef):
        coef = coef if int(coef) != coef else int(coef)
        if operator != "+":
            return f"{operator}"
        if coef == 1:
            return f"{operator}"
        if coef < 0:
            return f"{coef}x"
        return f"{operator}{coef}x"
