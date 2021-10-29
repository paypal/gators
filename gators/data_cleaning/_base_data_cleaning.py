# License: Apache-2.0
from abc import ABC
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer
from ..util import util


class _BaseDataCleaning(Transformer, ABC):
    """Base data cleaning transformer."""

    def __init__(self):
        Transformer.__init__(self)
        self.columns_to_drop: List[str] = []
        self.columns_to_keep: List[str] = []
        self.idx_columns_to_keep = np.array([])

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame]
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataset.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Dataset without datetime columns.
        """
        self.check_dataframe(X)
        if len(self.columns):
            return X.drop(self.columns, axis=1)
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
        return X[:, self.idx_columns_to_keep]

    @staticmethod
    def get_idx_columns_to_keep(
        columns: List[str], columns_to_drop: List[str]
    ) -> np.array:
        """Get the column indices to keep.

        Parameters
        ----------
        columns : List[str]
            List of columns of a dataset.
        columns_to_drop : List[str]
            List of columns to drop.

        Returns
        -------
        np.array:
            Column indices to keep.
        """
        idx_columns_to_keep = util.exclude_idx_columns(
            columns=columns,
            excluded_columns=columns_to_drop,
        )
        return idx_columns_to_keep
