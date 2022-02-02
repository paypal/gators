# License: Apache-2.0
from typing import List

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning

from gators import DataFrame, Series


class DropLowCardinality(_BaseDataCleaning):
    """Drop the categorical columns having a low cardinality.

    Parameters
    ----------
    min_categories : int
        Min categories allowed.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.data_cleaning import DropLowCardinality
    >>> obj = DropLowCardinality(min_categories=2)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A
    0  a
    1  b
    2  c

    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'd']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a'],
           ['b'],
           ['c']], dtype=object)
    """

    def __init__(self, min_categories: int):
        if not isinstance(min_categories, int):
            raise TypeError("`min_categories` should be an int.")
        _BaseDataCleaning.__init__(self)
        self.min_categories: int = min_categories

    def fit(self, X: DataFrame, y: Series = None) -> "DropLowCardinality":
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
        self : DropLowCardinality
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = self.get_columns_to_drop(X=X, min_categories=self.min_categories)
        self.columns_to_keep = util.exclude_columns(
            columns=X.columns,
            excluded_columns=self.columns,
        )
        self.idx_columns_to_keep = self.get_idx_columns_to_keep(
            columns=X.columns, columns_to_drop=self.columns
        )
        return self

    @staticmethod
    def get_columns_to_drop(X: DataFrame, min_categories: float) -> List[str]:
        """Get the list of the column names to remove.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame.
        min_categories : int
            Min categories allowed.

        Returns
        -------
        List[str]
            List of the column names to drop.
        """
        object_columns = util.get_datatype_columns(X, object)
        if not object_columns:
            return []
        X_nunique = util.get_function(X).to_pandas(
            util.get_function(X).melt(X).groupby("variable")["value"].nunique()
        )
        mask_columns = X_nunique < min_categories
        columns_to_drop = mask_columns[mask_columns].index
        return list(columns_to_drop)
