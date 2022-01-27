from itertools import chain, combinations
from typing import List

import numpy as np

from feature_gen import polynomial_object
from gators.transformers import Transformer

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class PolynomialObjectFeatures(Transformer):
    """Create new columns based on object columns addition.


    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    degree : int, default = 2
        The degree of polynomial. The default of degree of 2
        will produce A * A, B * B, and A  * B from features A and B.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.feature_generation import PolynomialObjectFeatures
    >>> obj = PolynomialObjectFeatures(columns=['A', 'B', 'C'], degree=3)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [None, 'b', 'c'], 'B': ['z', 'a', 'a'], 'C': ['c', 'd', 'd']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [None, 'b', 'c'], 'B': ['z', 'a', 'a'], 'C': ['c', 'd', 'd']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [None, 'b', 'c'], 'B': ['z', 'a', 'a'], 'C': ['c', 'd', 'd']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
          A  B  C A__B A__C B__C A__B__C
    0  None  z  c    z    c   zc      zc
    1     b  a  d   ba   bd   ad     bad
    2     c  a  d   ca   cd   ad     cad

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame({'A': [None, 'b', 'c'], 'B': ['z', 'a', 'a'], 'C': ['c', 'd', 'd']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[None, 'z', 'c', 'z', 'c', 'zc', 'zc'],
           ['b', 'a', 'd', 'ba', 'bd', 'ad', 'bad'],
           ['c', 'a', 'd', 'ca', 'cd', 'ad', 'cad']], dtype=object)
    """

    def __init__(
        self,
        columns: List[str],
        degree=2,
    ):
        self.degree = degree
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if not isinstance(degree, int):
            raise TypeError("`degree` should be an int.")
        if degree < 1:
            raise ValueError("`degree` should be at least 2.")
        self.combinations = list(
            map(
                list,
                chain.from_iterable(
                    combinations(columns, r=r) for r in range(2, self.degree + 1)
                ),
            )
        )
        column_names = [
            "__".join(map(str, combination)) for combination in self.combinations
        ]
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )

    def fit(self, X: DataFrame, y: Series = None) -> "PolynomialObjectFeatures":
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
            y (np.ndarray, optional): labels. Defaults to None.

        Returns
        -------
        self : PolynomialObjectFeatures
            Instance of itself.
        """
        self.check_dataframe(X)
        if self.degree == 1:
            return self
        columns = list(X.columns)
        self.combinations = list(
            map(
                list,
                chain.from_iterable(
                    combinations(self.columns, r=r) for r in range(2, self.degree + 1)
                ),
            )
        )

        self.combinations_np = [
            list(util.get_idx_columns(columns, cols)) for cols in self.combinations
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
        self.check_dataframe(X)
        if self.degree == 1:
            return X
        for combi, name in zip(self.combinations, self.column_names):
            X[name] = X[combi[0]].fillna("")
            X[name] += X["__".join(combi[1:])].fillna("")
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
        return polynomial_object(X, self.combinations_np, self.degree)
