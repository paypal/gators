# License: Apache-2.0
from typing import List

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning

from gators import DataFrame, Series


class DropHighNaNRatio(_BaseDataCleaning):
    """Drop the columns having a large NaN values ratio.

    Parameters
    ----------
    max_ratio : float
        Max nan ratio allowed.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.data_cleaning import DropHighNaNRatio
    >>> obj = DropHighNaNRatio(max_ratio=0.5)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    >>> X = pd.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1],
           [2],
           [3]], dtype=object)
    """

    def __init__(self, max_ratio: float):
        if not isinstance(max_ratio, float):
            raise TypeError("`max_ratio` should be a float.")
        _BaseDataCleaning.__init__(self)
        self.max_ratio = max_ratio

    def fit(self, X: DataFrame, y: Series = None) -> "DropHighNaNRatio":
        """Fit the transformer on the dataframe X.

        Get the list of column names to remove and the array of
           indices to be kept.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
           Target values.

        Returns
        -------
        self : DropHighNaNRatio
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = self.get_columns_to_drop(X=X, max_ratio=self.max_ratio)
        self.columns_to_keep = util.exclude_columns(
            columns=list(X.columns), excluded_columns=self.columns
        )
        self.idx_columns_to_keep = self.get_idx_columns_to_keep(
            columns=X.columns, columns_to_drop=self.columns
        )
        return self

    @staticmethod
    def get_columns_to_drop(X: DataFrame, max_ratio: float) -> List[str]:
        """Get  the list of column names to drop.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        max_ratio : float
            Max nan ratio allowed.

        Returns
        -------
        List[str]
            List of column names to drop.
        """
        mask_columns = util.get_function(X).to_pandas(X.isnull().mean()) > max_ratio
        columns_to_drop = list(mask_columns[mask_columns].index)
        return columns_to_drop
