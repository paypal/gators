# License: Apache-2.0
from typing import List

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning

from gators import DataFrame, Series


class DropHighCardinality(_BaseDataCleaning):
    """Drop the categorical columns having a large cardinality.

    Parameters
    ----------
    max_categories : int
        Maximum number of categories allowed.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.data_cleaning import DropHighCardinality
    >>> obj = DropHighCardinality(max_categories=2)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       B
    0  d
    1  d
    2  e

    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['d'],
           ['d'],
           ['e']], dtype=object)
    """

    def __init__(self, max_categories: int):
        if not isinstance(max_categories, int):
            raise TypeError("`max_categories` should be an int.")
        _BaseDataCleaning.__init__(self)
        self.max_categories = max_categories

    def fit(self, X: DataFrame, y: Series = None) -> "DropHighCardinality":
        """Fit the transformer on the dataframe `X`.

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
        self : DropHighCardinality
            Instance of itself.
        """
        self.check_dataframe(X)
        object_columns = util.get_datatype_columns(X, object)
        self.columns = self.get_columns_to_drop(
            X=X[object_columns], max_categories=self.max_categories
        )
        self.columns_to_keep = util.exclude_columns(
            columns=X.columns,
            excluded_columns=self.columns,
        )
        self.idx_columns_to_keep = self.get_idx_columns_to_keep(
            columns=X.columns,
            columns_to_drop=self.columns,
        )
        return self

    @staticmethod
    def get_columns_to_drop(X: DataFrame, max_categories: int) -> List[str]:
        """Get the names of the columns to drop.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe.
        max_categories : int
            Maximum number of categories allowed.

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
        mask_columns = X_nunique > max_categories
        columns_to_drop = X_nunique[mask_columns].index
        return list(columns_to_drop.to_numpy())
