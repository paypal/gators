from itertools import chain, combinations, combinations_with_replacement
from typing import List

import numpy as np

from feature_gen import polynomial
from gators.transformers import Transformer

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class PolynomialFeatures(Transformer):
    """Create new columns based on columns multiplication.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    degree : int, default = 2
        The degree of polynomial. The default of degree of 2
        will produce A * A, B * B, and A  * B from features A and B.
    interaction_only : bool, default = False
        Allows to keep only interaction terms.
        If true, only A * B will be produced from features A and B.
    dtype : type, default np.float64
        Numpy dtype of the output data.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.feature_generation import PolynomialFeatures
    >>> obj = PolynomialFeatures(columns=['A', 'B'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B    C  A__x__A  A__x__B  B__x__B
    0  0.0  1.0  2.0      0.0      0.0      1.0
    1  3.0  4.0  5.0      9.0     12.0     16.0
    2  6.0  7.0  8.0     36.0     42.0     49.0

    >>> X = pd.DataFrame(
    ... {'A': [0.0, 3.0, 6.0], 'B': [1.0, 4.0, 7.0], 'C': [2.0, 5.0, 8.0]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.,  1.,  2.,  0.,  0.,  1.],
           [ 3.,  4.,  5.,  9., 12., 16.],
           [ 6.,  7.,  8., 36., 42., 49.]])
    """

    def __init__(
        self,
        columns: List[str],
        degree=2,
        interaction_only=False,
    ):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if (not isinstance(degree, int)) or (degree < 1):
            raise TypeError("`degree` should be a positive int.")
        if not isinstance(interaction_only, bool):
            raise TypeError("`interaction_only` should be a bool.")
        if interaction_only == True and len(columns) == 1:
            raise ValueError("Cannot create interaction only terms from one column.")
        self.interaction_only = interaction_only
        self.columns = columns
        self.degree = degree
        self.method = (
            combinations if interaction_only else combinations_with_replacement
        )
        self.combinations = list(
            map(
                list,
                chain.from_iterable(
                    self.method(columns, e) for e in range(2, self.degree + 1)
                ),
            )
        )
        column_names = [
            "__x__".join(map(str, combination)) for combination in self.combinations
        ]
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )

    def fit(self, X: DataFrame, y: Series = None) -> "PolynomialFeatures":
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
            y (np.ndarray, optional): labels. Defaults to None.

        Returns
        -------
        self : PolynomialFeatures
            Instance of itself.
        """
        self.check_dataframe(X)
        if self.degree == 1:
            return self
        self.idx_subarray = util.get_idx_columns(X.columns, self.columns)
        self.combinations_np = [
            list(util.get_idx_columns(self.columns, cols)) for cols in self.combinations
        ]
        for combi in self.combinations_np:
            combi.extend([-1 for _ in range(self.degree - len(combi))])
        self.combinations_np = np.array(self.combinations_np)
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
        if self.degree == 1:
            return X
        self.check_dataframe(X)
        for combi, name in zip(self.combinations, self.column_names):
            X[name] = X[combi[0]]
            for c in combi[1:]:
                X[name] *= X[c]
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
        if self.degree == 1:
            return X
        X_new = polynomial(
            X[:, self.idx_subarray].astype(np.float64),
            self.combinations_np,
            self.degree,
        )
        return np.concatenate((X, X_new), axis=1)
