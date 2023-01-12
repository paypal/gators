# License: Apache-2.0
from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np


from encoder import encoder
from encoder import encoder_new
from ..util import util
from ..transformers.transformer import (
    Numeric_DTYPES,
    PRINT_Numeric_DTYPES,
    Transformer,
)

from gators import DataFrame


class _BaseEncoder(Transformer):
    """Base encoder transformer class.

    Parameters
    ----------
    inplace : bool.
        If True, perform the encoding inplace.
    """

    def __init__(self, inplace):
        if not isinstance(inplace, bool):
            raise TypeError(f"`inplace` should be a bool.")
        Transformer.__init__(self)
        self.inplace = inplace
        self.columns = []
        self.idx_columns: np.ndarray = np.array([])
        self.num_categories_vec = np.array([])
        self.values_vec = np.array([])
        self.encoded_values_vec = np.array([])
        self.mapping: Dict[str, Dict[str, float]] = {}

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

        if self.inplace:
            X = util.get_function(X).replace(X, self.mapping)
            X = util.get_function(X).to_numeric(X, columns=self.columns)
            return X
        X_encoded = util.get_function(X).replace(X.copy(), self.mapping)[self.columns]
        X_encoded = X_encoded.rename(columns=dict(zip(self.columns, self.column_names)))
        X_encoded = util.get_function(X).to_numeric(
            X_encoded, columns=self.column_names
        )
        X = util.get_function(X).join(X, X_encoded)

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
            Encoded array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X

        X_encoded = encoder_new(
            X[:, self.idx_columns],
            self.num_categories_vec,
            self.values_vec,
            self.encoded_values_vec,
        )
        if self.inplace:
            X[:, self.idx_columns] = X_encoded
            return X.astype(float)
        return np.concatenate((X, X_encoded), axis=1)

    @staticmethod
    def decompose_mapping(
        mapping: Dict[str, Dict[str, float]],
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Decompose the mapping.

        Parameters
        ----------
        mapping : Dict[str, Dict[str, float]]
            The dictionary keys are the categorical columns,
            the keys are the mapping itself for the assocaited column.

        Returns
        -------
        columns : List[float]
            List of columns.

        values_vec : np.ndarray
            Values to encode.

        encoded_values_vec : np.ndarray
            Values used to encode.
        """
        columns = list(mapping.keys())
        n_columns = len(columns)
        max_categories = max([len(m) for m in mapping.values()])
        encoded_values_vec = np.zeros((n_columns, max_categories))
        values_vec = np.zeros((n_columns, max_categories), dtype=object)
        for i, c in enumerate(columns):
            mapping_col = mapping[c]
            n_values = len(mapping_col)
            encoded_values_vec[i, :n_values] = np.array(list(mapping_col.values()))
            values_vec[i, :n_values] = np.array(list(mapping_col.keys()))
        return columns, values_vec, encoded_values_vec

    def display_mapping(self, cmap: Union[str, "colormap"], k=5, decimal=2, title=""):
        """Display the encoder mapping in a jupyter notebook.
        Parameters
        ----------
        cmap : Union[str, 'colormap']
            Matplotlib colormap.
        k : int, default 5.
            Number of mappings displayed.
        decimals : int, default 2.
            Number of decimal places to use.
        title : str, default ''.
            Plot title.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        if not isinstance(decimal, int) or decimal < 1:
            raise TypeError(f"`decimal` should be a positive int.")
        if not isinstance(k, int) or k < 1:
            raise TypeError(f"`k` should be a positive int.")
        if not isinstance(title, str):
            raise TypeError(f"`title` should be a str.")

        mapping = pd.DataFrame(self.mapping)
        vmin = mapping.min().min()
        vmax = mapping.max().max()

        cols = mapping.max().sort_values(ascending=False).index
        for c in cols[:k]:
            encoder_mapping_col = (
                mapping[[c]].dropna().sort_values(c, ascending=False).round(decimal)
            )
            x, y = 0.8 * len(encoder_mapping_col) / 1.62, 0.8 * len(encoder_mapping_col)
            _, ax = plt.subplots(figsize=(x, y))
            sns.heatmap(
                encoder_mapping_col,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                annot=True,
                cbar=False,
            )
            _ = ax.set_title(title)
            _ = ax.set_ylabel(None)
            _ = ax.set_ylabel(c)
