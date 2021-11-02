# License: Apache-2.0
from typing import List

import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class _BaseFeatureSelection(Transformer):
    """Base feature selection transformer class."""

    def __init__(self):
        self.feature_importances_ = pd.Series([], dtype=np.float64)
        self.selected_columns: List[str] = []
        self.idx_selected_columns: List[str] = []
        self.columns_to_drop: List[str] = []

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : np.ndarray
             Target values.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        columns_to_drop = [
            c for c in self.columns_to_drop if c in X.columns
        ]  # needed for dask
        if len(columns_to_drop):
            return X.drop(columns_to_drop, axis=1)
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
        self.idx_selected_columns.sort()
        return X[:, self.idx_selected_columns]
