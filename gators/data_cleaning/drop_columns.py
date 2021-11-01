# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import pandas as pd

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning


class DropColumns(_BaseDataCleaning):
    """Drop the columns given by the user.

    Parameters
    ----------
    columns : List[str]
        List of columns to drop.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropColumns
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = DropColumns(['B'])
    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropColumns
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = DropColumns(['B'])
    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropColumns
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = DropColumns(['B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1],
           [2],
           [3]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropColumns
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = DropColumns(['B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1],
           [2],
           [3]])

    """

    def __init__(self, columns):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")

        _BaseDataCleaning.__init__(self)
        self.columns = columns

    def fit(self, X: Union[pd.DataFrame, ks.DataFrame], y=None) -> "DropColumns":
        """Fit the transformer on the dataframe X.

        Get the list of column names to remove and the array of
          indices to be kept.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        DropColumns: Instance of itself.
        """
        self.check_dataframe(X)
        self.columns_to_keep = util.exclude_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        self.idx_columns_to_keep = util.exclude_idx_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        return self
