# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen_str import split_and_extract_str

from ._base_string_feature import _BaseStringFeature


class SplitExtract(_BaseStringFeature):
    """Create new columns based on split strings.

    The transformer applies two steps:

    * split each string element using the given value.
    * extract the string of the given split list element.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    str_split_vec : List[int]
        List of separators.

    idx_split_vec : List[int]
        List of split indices.

    column_names : List[int]
        List of new column names.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import SplitExtract
    >>> X = pd.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})
    >>> obj = SplitExtract(
    ...     columns=['A','A'], str_split_vec=['*', '*'], idx_split_vec=[0, 1])
    >>> obj.fit_transform(X)
          A  B A__split_by_*_idx_0 A__split_by_*_idx_1
    0  qw*e  1                  qw                   e
    1  a*qd  2                   a                  qd
    2  zxq*  3                 zxq

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import SplitExtract
    >>> X = ks.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})
    >>> obj = SplitExtract(
    ...     columns=['A','A'], str_split_vec=['*', '*'], idx_split_vec=[0, 1])
    >>> obj.fit_transform(X)
          A  B A__split_by_*_idx_0 A__split_by_*_idx_1
    0  qw*e  1                  qw                   e
    1  a*qd  2                   a                  qd
    2  zxq*  3                 zxq

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_str import SplitExtract
    >>> X = pd.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})
    >>> obj = SplitExtract(
    ...     columns=['A','A'], str_split_vec=['*', '*'], idx_split_vec=[0, 1])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qw*e', 1, 'qw', 'e'],
           ['a*qd', 2, 'a', 'qd'],
           ['zxq*', 3, 'zxq', '']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_str import SplitExtract
    >>> X = ks.DataFrame({'A': ['qw*e', 'a*qd', 'zxq*'], 'B': [1, 2, 3]})
    >>> obj = SplitExtract(
    ...     columns=['A','A'], str_split_vec=['*', '*'], idx_split_vec=[0, 1])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['qw*e', 1, 'qw', 'e'],
           ['a*qd', 2, 'a', 'qd'],
           ['zxq*', 3, 'zxq', '']], dtype=object)

    """

    def __init__(
        self,
        columns: List[str],
        str_split_vec: List[int],
        idx_split_vec: List[int],
        column_names: List[str] = None,
    ):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not isinstance(str_split_vec, list):
            raise TypeError("`str_split_vec` should be a list.")
        if len(columns) != len(str_split_vec):
            raise ValueError("Length of `columns` and `str_split_vec` should match.")
        if not isinstance(idx_split_vec, list):
            raise TypeError("`idx_split_vec` should be a list.")
        if len(columns) != len(idx_split_vec):
            raise ValueError("Length of `columns` and `idx_split_vec` should match.")
        if not column_names:
            column_names = [
                f"{col}__split_by_{split}_idx_{idx}"
                for col, split, idx in zip(columns, str_split_vec, idx_split_vec)
            ]
        _BaseStringFeature.__init__(self, columns, column_names)
        self.str_split_vec = np.array(str_split_vec, object)
        self.idx_split_vec = np.array(idx_split_vec, int)

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
        for col, idx, str_split, name in zip(
            self.columns, self.idx_split_vec, self.str_split_vec, self.column_names
        ):
            n = idx if idx > 0 else 1
            X.loc[:, name] = (
                X[col].str.split(str_split, n=n, expand=True)[idx].fillna("MISSING")
            )
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
        return split_and_extract_str(
            X, self.idx_columns, self.str_split_vec, self.idx_split_vec
        )
