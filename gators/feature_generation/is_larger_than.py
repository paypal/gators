# License: Apache-2.0
from typing import Dict, List

import numpy as np

from feature_gen import one_hot

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class IsLargerThan(_BaseFeatureGeneration):
    """Create new columns based on the following definition: X[binarized_col] = X[col] >= bound.

    Parameters
    ----------
    bounds_dict : Dict[str: List[float]].
        keys: columns, values: list of bounds.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import OneHot
    >>> bounds_dict = {'A': ['b', 'c'], 'B': ['z']}
    >>> obj = OneHot(bounds_dict=bounds_dict)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']}), npartitions=1)

    * `koalas` dataframes:

    >>> import pyspark.pandas as ps
    >>> X = ps.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A  B  A__onehot__b  A__onehot__c  B__onehot__z
    0  a  z         False         False          True
    1  b  a          True         False         False
    2  c  a         False          True         False

    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'z', False, False, True],
           ['b', 'a', True, False, False],
           ['c', 'a', False, True, False]], dtype=object)

    """

    def __init__(
        self, bounds_dict: Dict[str, List[float]], column_names: List[str] = None
    ):
        if not isinstance(bounds_dict, dict):
            raise TypeError("`bounds_dict` should be a dict.")
        if column_names is not None and not isinstance(
            column_names, (list, np.ndarray)
        ):
            raise TypeError("`column_names` should be None or a list.")
        self.bounds_dict = bounds_dict
        columns = list(set(bounds_dict.keys()))
        if not column_names:
            column_names = [
                f"{col}__{bound}_inf"
                for col, bounds in bounds_dict.items()
                for bound in bounds
            ]
        columns = [col for col, bounds in bounds_dict.items() for _ in bounds]
        n_bounds = sum(len(bound) for bound in bounds_dict.values())
        if column_names and n_bounds != len(column_names):
            raise ValueError(
                "Length of `clusters_dict` and `column_names` should match."
            )

        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )
        self.mapping = dict(
            zip(
                column_names,
                [
                    [col, bound]
                    for col, bounds in bounds_dict.items()
                    for bound in bounds
                ],
            )
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
        self : OneHot
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.bounds = np.array(
            [bound for bounds in self.bounds_dict.values() for bound in bounds]
        )
        cols_flatten = np.array(
            [col for col, bounds in self.bounds_dict.items() for _ in bounds]
        )
        self.idx_columns = util.get_idx_columns(X, cols_flatten)
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
        new_series_list = []
        for name, col, bound in zip(self.column_names, self.columns, self.bounds):
            dummy = X[col] >= bound
            new_series_list.append(dummy.rename(name))

        return util.get_function(X).concat(
            [X, util.get_function(X).concat(new_series_list, axis=1)],
            axis=1,
        )

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
        # return one_hot(X, self.idx_columns, self.bounds)
