# License: Apache-2.0
from typing import List

import numpy as np
import pandas as pd

from feature_gen_str import string_length

from ..util import util
from ._base_string_feature import _BaseStringFeature

pd.options.mode.chained_assignment = None


from gators import DataFrame, Series


class StringLength(_BaseStringFeature):
    """Create new columns based on the length of its elements.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_str import StringLength
    >>> obj = StringLength(columns=['A', 'B'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(
    ... pd.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B  A__length  B__length
    0  qwe    1        3.0        1.0
    1   as   22        2.0        2.0
    2       333        0.0        3.0

    >>> X = pd.DataFrame({'A': ['qwe', 'as', ''], 'B': [1, 22, 333]})
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

    def fit(self, X: DataFrame, y: Series = None) -> "StringLength":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

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

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)

        for name, col in zip(self.column_names, self.columns):
            X[name] = (
                util.get_function(X)
                .replace(X[col].fillna("").astype(str), {"nan": ""})
                .str.len()
                .astype(np.float64)
            )
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        return string_length(X, self.idx_columns)
