# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen_str import string_length

from ..util import util

from ._base_string_feature import _BaseStringFeature

pd.options.mode.chained_assignment = None


class StringLength(_BaseStringFeature):
    """Create new columns based on the length of its elements.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import StringLength
    >>> X = pd.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})
    >>> obj = StringLength(columns=['A','B'])
    >>> obj.fit_transform(X)
         A    B  A__length  B__length
    0  qwe    1        3.0        1.0
    1   as   22        2.0        2.0
    2       333        0.0        3.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import StringLength
    >>> X = ks.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})
    >>> obj = StringLength(columns=['A','B'])
    >>> obj.fit_transform(X)
         A    B  A__length  B__length
    0  qwe    1        3.0        1.0
    1   as   22        2.0        2.0
    2       333        0.0        3.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import StringLength
    >>> X = pd.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})
    >>> obj = StringLength(columns=['A','B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qwe', 1, 3.0, 1.0],
           ['as', 22, 2.0, 2.0],
           ['', 333, 0.0, 3.0]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import StringLength
    >>> X = ks.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})
    >>> obj = StringLength(columns=['A','B'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qwe', 1, 3.0, 1.0],
           ['as', 22, 2.0, 2.0],
           ['', 333, 0.0, 3.0]], dtype=object)


    """

    def __init__(self, columns: List[str], column_names: List[str] = None):
        if not column_names:
            column_names = [f"{col}__length" for col in columns]
        _BaseStringFeature.__init__(self, columns, column_names)

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "StringLength":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
        StringLength
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
        )
        return self

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame]
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Transformed dataframe.
        """
        self.check_dataframe(X)
        for col, name in zip(self.columns, self.column_names):
            X.loc[:, name] = (
                X.loc[:, col]
                .fillna("")
                .astype(str)
                .replace({"nan": ""})
                .str.len()
                .astype(np.float64)
            )
            pass
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.check_array(X)
        return string_length(X, self.idx_columns)
