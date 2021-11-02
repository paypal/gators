# License: Apache-2.0
# from abc import ABC
from typing import List

import numpy as np

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame


class _BaseDataCleaning(Transformer):
    """Base data cleaning transformer."""

    def __init__(self):
        Transformer.__init__(self)
        self.columns_to_drop: List[str] = []
        self.columns_to_keep: List[str] = []
        self.idx_columns_to_keep = np.array([])

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataset.

        Returns
        -------
        X : DataFrame
            Transformed dataset.
        """
        self.check_dataframe(X)
        self.columns_ = list(X.columns)
        if len(self.columns):
            return X.drop(self.columns, axis=1)
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
        if self.idx_columns_to_keep.size == 0:
            return np.array([]).reshape(0, X.shape[1])
        return X[:, self.idx_columns_to_keep]

    @staticmethod
    def get_idx_columns_to_keep(
        columns: List[str], columns_to_drop: List[str]
    ) -> np.array:
        """Get the column indices to keep.

        Parameters
        ----------
        theta_vec : List[float]
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
