# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import pandas as pd

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning


class DropDatatypeColumns(_BaseDataCleaning):
    """Drop the columns belonging to a given datatype.

    Parameters
    ----------
    dtype : type
        Colum datatype to drop.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropDatatypeColumns
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1., 1., 1.]})
    >>> obj = DropDatatypeColumns(float)
    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropDatatypeColumns
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1., 1., 1.]})
    >>> obj = DropDatatypeColumns(float)
    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.data_cleaning import DropDatatypeColumns
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1., 1., 1.]})
    >>> obj = DropDatatypeColumns(float)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.],
           [2.],
           [3.]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropDatatypeColumns
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1., 1., 1.]})
    >>> obj = DropDatatypeColumns(float)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.],
           [2.],
           [3.]])

    """

    def __init__(self, dtype: type):
        if not isinstance(dtype, type):
            raise TypeError("`dtype` should be a type.")
        _BaseDataCleaning.__init__(self)
        self.dtype = dtype

    def fit(
        self, X: Union[pd.DataFrame, ks.DataFrame], y=None
    ) -> "DropDatatypeColumns":
        """Fit the transformer on the dataframe X.

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
        DropDatatypeColumns: Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_datatype_columns(X, self.dtype)
        self.columns_to_keep = util.exclude_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        self.idx_columns_to_keep = util.exclude_idx_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        return self
