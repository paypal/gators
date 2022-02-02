# License: Apache-2.0
import numpy as np

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning

from gators import DataFrame, Series


class DropColumns(_BaseDataCleaning):
    """Drop the columns given by the user.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns to drop.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.data_cleaning import DropColumns
    >>> obj = DropColumns(columns=['B'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1],
           [2],
           [3]])
    """

    def __init__(self, columns):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        self.columns = columns
        _BaseDataCleaning.__init__(self)

    def fit(self, X: DataFrame, y: Series = None) -> "DropColumns":
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
        self : DropColumns
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns_to_keep = util.exclude_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        self.idx_columns_to_keep = util.exclude_idx_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        return self
