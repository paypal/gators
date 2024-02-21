# License: Apache-2.0
from typing import List, Dict
import numpy as np


from gators import DataFrame, Series
from encoder import encoder_new
from ..util import util
from ..transformers.transformer import Transformer

from gators import DataFrame


class _BaseEncoder(Transformer):
    """Base encoder transformer class.

    Parameters
    ----------
    inplace : bool.
        If True, perform the encoding inplace.
    """

    def __init__(self, columns: List[str], inplace: bool):
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        Transformer.__init__(self)
        self.inplace = inplace
        self.columns = columns
        self.idx_columns: np.ndarray = np.array([])
        self.num_categories_vec = np.array([])
        self.values_vec = np.array([])
        self.encoded_values_vec = np.array([])
        self.mapping: Dict[str, Dict[str, float]] = {}

    def fit(self, X: DataFrame, y: Series = None) -> "_BaseEncoder":
        """Fit the encoder.

        Parameters
        ----------
        X : DataFrame:
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        WOEEncoder:
            Instance of itself.
        """
        self.check_dataframe(X)
        self.set_columns(
            X=X, include=[bool, object, "string[pyarrow]"], suffix=self.suffix
        )
        if not self.columns:
            return self
        self.mapping = self.generate_mapping(X[self.columns], y)
        self.mapping = {
            k: {k: str(round(v, 4)) for k, v in v.items()}
            for k, v in self.mapping.items()
        }
        self.set_num_categories_vec()
        self.decompose_mapping()
        return self

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
            X[self.columns] = X[self.columns]
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

    def set_num_categories_vec(self):
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])

    def decompose_mapping(self):
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
        n_columns = len(self.columns)
        max_categories = max(len(m) for m in self.mapping.values())
        self.encoded_values_vec = np.zeros((n_columns, max_categories))
        self.values_vec = np.zeros((n_columns, max_categories), dtype=object)
        for i, c in enumerate(self.columns):
            mapping_col = self.mapping[c]
            n_values = len(mapping_col)
            self.encoded_values_vec[i, :n_values] = np.array(list(mapping_col.values()))
            self.values_vec[i, :n_values] = np.array(list(mapping_col.keys()))
