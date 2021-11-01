# License: Apache-2.0
from typing import Any, Collection, Dict, List, Tuple, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from encoder import encoder

from ..transformers.transformer import (
    NUMERICS_DTYPES,
    PRINT_NUMERICS_DTYPES,
    Transformer,
)


class _BaseEncoder(Transformer):
    """Base encoder transformer class.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.
    """

    def __init__(self, dtype=np.float64):
        if dtype not in NUMERICS_DTYPES:
            raise TypeError(f"`dtype` should be a dtype from {PRINT_NUMERICS_DTYPES}.")
        Transformer.__init__(self)
        self.dtype = dtype
        self.columns = []
        self.idx_columns: np.ndarray = np.array([])
        self.num_categories_vec = np.array([])
        self.values_vec = np.array([])
        self.encoded_values_vec = np.array([])
        self.mapping: Dict[str, Dict[str, float]] = {}

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
        return X.replace(self.mapping).astype(self.dtype)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the input array.

        Parameters
        ----------
        X  : np.ndarray
            Input array.
        Returns
        -------
        np.ndarray: Encoded array.
        """
        self.check_array(X)
        if len(self.idx_columns) == 0:
            return X.astype(self.dtype)
        return encoder(
            X,
            self.num_categories_vec,
            self.values_vec,
            self.encoded_values_vec,
            self.idx_columns,
        ).astype(self.dtype)

    @staticmethod
    def decompose_mapping(
        mapping: List[Dict[str, Collection[Any]]],
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Decompose the mapping.

        Parameters
        ----------
        mapping List[Dict[str, Collection[Any]]]:
            Mapping obtained from the categorical encoder package.
        Returns
        -------
        Tuple[List[str], np.ndarray, np.ndarray]
            Decomposed mapping.
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
