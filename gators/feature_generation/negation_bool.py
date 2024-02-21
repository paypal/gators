# License: Apache-2.0
from typing import Dict, List

import numpy as np

from feature_gen import one_hot

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class NegationBool(_BaseFeatureGeneration):
    """Create new columns based on the negation operator.

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

    def __init__(self, columns: List[str], column_names: List[str] = None):
        if not column_names:
            column_names = [f"{col}__NOT" for col in columns]
        if len(columns) != len(column_names):
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
        self : OneHot
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns = util.get_idx_columns(X, self.columns)
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
        X_dummy = ~(X[self.columns].astype(bool)).rename(
            columns=dict(zip(self.columns, self.column_names))
        )
        return util.get_function(X).concat([X, X_dummy], axis=1)

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
