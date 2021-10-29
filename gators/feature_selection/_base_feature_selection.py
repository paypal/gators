# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer
from ..util import util


class _BaseFeatureSelection(Transformer):
    """Base feature selection transformer class.

    Parameters
    ----------
    columns: List[str]
        List of columns to drop.
    selected_columns : List[str]
        List of selected columns.
    feature_importances_ : pd.Series
        Feature importance.

    """

    def __init__(self):
        self.feature_importances_ = pd.Series([], dtype=np.float64)
        self.selected_columns: List[str] = []
        self.idx_selected_columns: List[str] = []
        self.columns_to_drop: List[str] = []

    def transform(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : np.ndarray
             Labels.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Transformed dataframe.
        """
        self.check_dataframe(X)
        if len(self.columns_to_drop):
            return X.drop(self.columns_to_drop, axis=1)
        else:
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
        self.idx_selected_columns.sort()
        return X[:, self.idx_selected_columns]
