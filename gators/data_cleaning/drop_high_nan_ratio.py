# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import pandas as pd

from ..util import util
from ._base_data_cleaning import _BaseDataCleaning


class DropHighNaNRatio(_BaseDataCleaning):
    """Drop the columns having a large NaN values ratio.

    Parameters
    ----------
    max_ratio : float
        Max nan ratio allowed.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import numpy as np
    >>> import pandas as pd
    >>> from gators.data_cleaning import DropHighNaNRatio
    >>> X = pd.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})
    >>> obj = DropHighNaNRatio(max_ratio=0.5)
    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    * fit & transform with `koalas`

    >>> import numpy as np
    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropHighNaNRatio
    >>> X = ks.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})
    >>> obj = DropHighNaNRatio(max_ratio=0.5)
    >>> obj.fit_transform(X)
       A
    0  1
    1  2
    2  3

    * fit with `pandas` & transform with `NumPy`

    >>> import numpy as np
    >>> import pandas as pd
    >>> from gators.data_cleaning import DropHighNaNRatio
    >>> X = pd.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})
    >>> obj = DropHighNaNRatio(max_ratio=0.5)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1],
           [2],
           [3]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import numpy as np
    >>> import databricks.koalas as ks
    >>> from gators.data_cleaning import DropHighNaNRatio
    >>> X = ks.DataFrame(
    ... {'A': [1, 2, 3], 'B': ['1', None, None], 'C': [1., np.nan, np.nan]})
    >>> obj = DropHighNaNRatio(max_ratio=0.5)
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

    def fit(self, X: Union[pd.DataFrame, ks.DataFrame], y=None) -> "DropHighNaNRatio":
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
        DropHighNaNRatio: Instance of itself.
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
    def get_columns_to_drop(
        X: Union[pd.DataFrame, ks.DataFrame], max_ratio: float
    ) -> List[str]:
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
        mask_columns = X.isnull().mean() > max_ratio
        columns_to_drop = mask_columns[mask_columns].index
        if isinstance(columns_to_drop, ks.indexes.Index):
            columns_to_drop = columns_to_drop.to_pandas()
        return columns_to_drop
