# License: Apache-2.0
import numpy as np

from scaler import scaler

from ..transformers.transformer import Transformer
from gators import DataFrame, Series


class _BaseScaler(Transformer):
    """Base scaler transformer class.

    Parameters
    ----------
    inplace : bool.
        If True, perform the encoding inplace.
    """

    def __init__(self, inplace: bool):
        self.X_offset: DataFrame = None
        self.X_scale: DataFrame = None
        self.X_offset_np = np.array([])
        self.X_scale_np = np.array([])
        self.inplace = inplace

    def transform(self, X):
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
        X[self.columns] = X[self.columns].astype(float)
        for col, name in zip(self.columns, self.column_names):
            X[name] = (X[col] - self.X_offset[col]) * self.X_scale[col]
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X :np.ndarray:
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        X_scale = scaler(X.astype(float), self.X_offset_np, self.X_scale_np)
        if self.inplace:
            X[:, self.idx_columns] = X_scale
            return X
        return np.concatenate((X, X_scale), axis=1)
