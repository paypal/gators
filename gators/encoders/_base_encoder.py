# License: Apache-2.0
from typing import List, Tuple, Dict
import numpy as np
from encoder import encoder

from ..util import util
from ..transformers.transformer import (
    NUMERICS_DTYPES,
    PRINT_NUMERICS_DTYPES,
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
        self.dtypes_ = X.dtypes
        if self.inplace:
            X = util.get_function(X).replace(X, self.mapping)
            X = util.get_function(X).to_numeric(X, columns=self.columns)
            self.dtypes_ = X.dtypes
            return X
        X_encoded = util.get_function(X).replace(X.copy(), self.mapping)[self.columns]
        X_encoded = X_encoded.rename(columns=dict(zip(self.columns, self.column_names)))
        X_encoded = util.get_function(X).to_numeric(
            X_encoded, columns=self.column_names
        )
        X = util.get_function(X).join(X, X_encoded)
        self.dtypes_ = X.dtypes
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
        X_encoded = encoder(
            X.copy(),
            self.num_categories_vec,
            self.values_vec,
            self.encoded_values_vec,
            self.idx_columns,
        )
        if self.inplace:
            return X_encoded.astype(float)
        return np.concatenate((X, X_encoded[:, self.idx_columns]), axis=1)

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
