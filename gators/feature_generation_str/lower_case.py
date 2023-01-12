# License: Apache-2.0
from typing import List

import numpy as np

from feature_gen_str import lower_case

from ..util import util
from ._base_string_feature import _BaseStringFeature

from gators import DataFrame, Series


class LowerCase(_BaseStringFeature):
    """Convert the selected columns to lower case.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_str import LowerCase
    >>> obj = LowerCase(columns=['A', 'B'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(
    ... pd.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A     B
    0  abc   abc
    1   ab    ab
    2       None

    >>> X = pd.DataFrame({'A': ['abC', 'Ab', ''], 'B': ['ABc', 'aB', None]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['abc', 'abc'],
           ['ab', 'ab'],
           ['', None]], dtype=object)
    """

    def __init__(self, columns: List[str], inplace: bool = True):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        self.columns = columns
        self.inplace = inplace
        self.column_names = self.get_column_names(self.inplace, self.columns, "lower")

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
            X[name] = util.get_function(X).replace(
                X[col].astype(str).str.lower(), {"none": None, "nan": None}
            )

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
        if self.inplace:
            X[:, self.idx_columns] = lower_case(X[:, self.idx_columns])
            return X
        else:
            X_upper = lower_case(X[:, self.idx_columns])
            return np.concatenate((X, X_upper), axis=1)
