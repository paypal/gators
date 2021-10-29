# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import pandas as pd

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning


class DropHighCardinality(_BaseDataCleaning):
    """Drop the categorical columns having a large cardinality.

    Parameters
    ----------
    max_categories : int
        Maximum number of categories allowed.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropHighCardinality
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})
    >>> obj = DropHighCardinality(max_categories=2)
    >>> obj.fit_transform(X)
       B
    0  d
    1  d
    2  e

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropHighCardinality
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})
    >>> obj = DropHighCardinality(max_categories=2)
    >>> obj.fit_transform(X)
       B
    0  d
    1  d
    2  e

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropHighCardinality
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})
    >>> obj = DropHighCardinality(max_categories=2)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['d'],
           ['d'],
           ['e']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropHighCardinality
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'd', 'e']})
    >>> obj = DropHighCardinality(max_categories=2)
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

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "DropHighCardinality":
        """Fit the transformer on the dataframe `X`.

        Get the list of column names to remove and the array of
           indices to be kept.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : None
           None

        Returns
        -------
            DropHighCardinality: Instance of itself.
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
    def get_columns_to_drop(
        X: Union[pd.DataFrame, ks.DataFrame], max_categories: int
    ) -> List[str]:
        """Get the column names to drop.

        Parameters
        ----------
        X_nunique : pd.DataFrame
            Input dataframe.
        max_categories : int
            Maximum number of categories allowed.

        Returns
        -------
        List[str]
            List of column names to drop.
        """
        object_columns = util.get_datatype_columns(X, object)
        if not object_columns:
            return []
        if isinstance(X, pd.DataFrame):
            X_nunique = X[object_columns].nunique()
        else:
            X_nunique = X[object_columns].nunique(approx=True)
        mask_columns = X_nunique > max_categories
        columns_to_drop = X_nunique[mask_columns].index
        return list(columns_to_drop.to_numpy())
