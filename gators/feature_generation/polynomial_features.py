from itertools import chain, combinations, combinations_with_replacement
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import polynomial
from gators.transformers import Transformer

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration


class PolynomialFeatures(Transformer):
    """Create new columns based on columns multiplication.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `PolynomialFeatures`.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    degree : int, default = 2
        The degree of polynomial. The default of degree of 2
        will produce A * A, B * B, and A  * B from features A and B.
    interaction_only : bool, default = False
        Allows to keep only interaction terms.
        If true, only A * B will be produced from features A and B.
    dtype : type, default to np.float64
        Numpy dtype of the output data.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation import PolynomialFeatures
    >>> X = pd.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})
    >>> obj = PolynomialFeatures(columns=['A', 'B'])
    >>> obj.fit_transform(X)
         A    B    C  A__x__A  A__x__B  B__x__B
    0  0.0  1.0  2.0      0.0      0.0      1.0
    1  3.0  4.0  5.0      9.0     12.0     16.0
    2  6.0  7.0  8.0     36.0     42.0     49.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import PolynomialFeatures
    >>> X = ks.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})
    >>> obj = PolynomialFeatures(
    ... columns=['A', 'B', 'C'], degree=3, interaction_only=True)
    >>> obj.fit_transform(X)
         A    B    C  A__x__B  A__x__C  B__x__C  A__x__B__x__C
    0  0.0  1.0  2.0      0.0      0.0      2.0            0.0
    1  3.0  4.0  5.0     12.0     15.0     20.0           60.0
    2  6.0  7.0  8.0     42.0     48.0     56.0          336.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import PolynomialFeatures
    >>> X = pd.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})
    >>> obj = PolynomialFeatures(
    ... columns=['A', 'B', 'C'], degree=2, interaction_only=True)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.,  1.,  2.,  0.,  0.,  2.],
           [ 3.,  4.,  5., 12., 15., 20.],
           [ 6.,  7.,  8., 42., 48., 56.]])

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import PolynomialFeatures
    >>> X = ks.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})
    >>> obj = PolynomialFeatures(
    ... columns=['A', 'B', 'C'], degree=2, interaction_only=True)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.,  1.,  2.,  0.,  0.,  2.],
           [ 3.,  4.,  5., 12., 15., 20.],
           [ 6.,  7.,  8., 42., 48., 56.]])

    """

    def __init__(
        self,
        columns: List[str],
        degree=2,
        interaction_only=False,
        dtype: type = np.float64,
    ):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if not isinstance(degree, int):
            raise TypeError("`degree` should be an int.")
        if degree < 2:
            raise ValueError("`degree` should be at least 2.")
        if not isinstance(interaction_only, bool):
            raise TypeError("`interaction_only` should be a bool.")
        if interaction_only == True and len(columns) == 1:
            raise ValueError(
                "Cannot create interaction only terms from single feature."
            )
        self.check_datatype(dtype, [np.float32, np.float64])

        self.degree = degree
        self.method = (
            combinations if interaction_only else combinations_with_replacement
        )
        self.combinations = list(
            map(
                list,
                chain.from_iterable(
                    self.method(columns, self.degree)
                    for self.degree in range(self.degree + 1)
                ),
            )
        )
        self.combinations = [c for c in self.combinations if len(c) >= 2]
        column_names = [
            "__x__".join(map(str, combination)) for combination in self.combinations
        ]
        column_mapping = dict(zip(column_names, map(list, self.combinations)))
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
    ) -> "PolynomialFeatures":
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
            y (np.ndarray, optional): labels. Defaults to None.

        Returns
        -------
            PolynomialFeatures: Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.dtype = X[self.columns].dtypes.unique()[0]
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
        )
        self.n_rows = X[self.columns].shape[0]
        self.n_cols = X[self.columns].shape[1]
        self.combinations_np = list(
            map(
                list,
                chain.from_iterable(
                    self.method(self.idx_columns, self.degree)
                    for self.degree in range(self.degree + 1)
                ),
            )
        )
        self.combinations_np = [c for c in self.combinations_np if len(c) >= 2]
        for combo in self.combinations_np:
            combo.extend([-1 for _ in range(self.degree - len(combo))])
        self.combinations_np = np.array(self.combinations_np)
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
            for combi, name in zip(self.combinations, self.column_names):
                X[name] = X[combi].prod(axis=1)
            X[self.column_names] = X[self.column_names].astype(self.dtype)
            return X
        for combi, name in zip(self.combinations, self.column_names):
            dummy = X[combi[0]] * X["__x__".join(combi[1:])]
            X = X.assign(dummy=dummy).rename(columns={"dummy": name})
        X[self.column_names] = X[self.column_names].astype(self.dtype)
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
        return polynomial(X, self.combinations_np, self.degree, self.dtype)
