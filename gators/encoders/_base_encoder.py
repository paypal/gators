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
    dtype : type, default np.float64.
        Numerical datatype of the output data.
    """

    def __init__(self, add_missing_categories, dtype=np.float64):
        if not isinstance(add_missing_categories, bool):
            raise TypeError("`add_missing_categories` should be a bool.")
        if dtype not in NUMERICS_DTYPES:
            raise TypeError(f"`dtype` should be a dtype from {PRINT_NUMERICS_DTYPES}.")
        Transformer.__init__(self)
        self.add_missing_categories = add_missing_categories
        self.dtype = dtype
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
        self.columns_ = list(X.columns)
        return util.get_function(X).replace(X, self.mapping).astype(self.dtype)

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

    @staticmethod
    def clean_mapping(
        mapping: Dict[str, Dict[str, List[float]]], add_missing_categories: bool
    ) -> Dict[str, Dict[str, List[float]]]:
        """Clean mapping.

        Parameters
        ----------
        mapping : Dict[str, Dict[str, List[float]]]
            Map the categorical values to the encoded ones.
        add_missing_categories: bool
            If True, add the columns 'OTHERS' and 'MISSING'
            to the mapping even if the categories are not
            present in the data.
        Returns
        -------
        Dict[str, Dict[str, List[float]]]
            Cleaned mapping
        """
        mapping = {
            col: {k: v for k, v in mapping[col].items() if v == v}
            for col in mapping.keys()
        }
        for m in mapping.values():
            if add_missing_categories and "OTHERS" not in m:
                m["OTHERS"] = 0.0
            if add_missing_categories and "MISSING" not in m:
                m["MISSING"] = 0.0
        return mapping
