# License: Apache-2.0
from typing import Dict, List

import numpy as np

from ..util import util

from .is_larger_than import IsLargerThan

from gators import DataFrame


class IsSmallerThan(IsLargerThan):
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
        IsLargerThan.__init__(self, bounds_dict=bounds_dict, column_names=column_names)
        if not column_names:
            self.column_names = [
                f"{col}__-inf_{bound}"
                for col, bounds in bounds_dict.items()
                for bound in bounds
            ]
        else:
            self.column_names = column_names
        self.columns = [col for col, bounds in bounds_dict.items() for _ in bounds]
        self.mapping = dict(
            zip(
                self.column_names,
                [
                    [col, bound]
                    for col, bounds in bounds_dict.items()
                    for bound in bounds
                ],
            )
        )

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
            dummy = X[col] < bound
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
