# License: Apache-2.0
import numpy as np

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning

from gators import DataFrame, Series


class KeepColumns(_BaseDataCleaning):
    """Drop the columns which are not given by the user.

    Parameters
    ----------
    columns_to_keep : List[str]
        List of columns.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.data_cleaning import KeepColumns
    >>> obj = KeepColumns(columns_to_keep=['A'])

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

    def __init__(self, columns_to_keep):
        if not isinstance(columns_to_keep, (list, np.ndarray)):
            raise TypeError("`columns_to_keep` should be a list.")
        _BaseDataCleaning.__init__(self)
        self.columns_to_keep = columns_to_keep

    def fit(self, X: DataFrame, y: Series = None) -> "KeepColumns":
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
        self : KeepColumns
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.exclude_columns(
            columns=X.columns, excluded_columns=self.columns_to_keep
        )
        self.idx_columns_to_keep = util.exclude_idx_columns(
            columns=X.columns, excluded_columns=self.columns
        )
        return self
