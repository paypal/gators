# License: Apache-2.0
from typing import Dict

import pandas as pd
import numpy as np

from binning import binning_new

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class _BaseBinning(Transformer):
    """Base binning transformer class.

    Parameters
    ----------
    n_bins : int
        Number of bins to use.
    inplace : bool
        If False, return the dataframe with the new binned columns
        with the names *column_name__bin*). Otherwise,
        return the dataframe with the existing binned columns.
    """

    def __init__(self, n_bins: int, inplace: bool):
        if (not isinstance(n_bins, int)) or (n_bins <= 0):
            raise TypeError("`n_bins` should be a positive int.")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        Transformer.__init__(self)
        self.n_bins = n_bins
        self.inplace = inplace
        self.base_columns = []
        self.columns = []
        self.column_names = []
        self.idx_columns = np.array([])
        self.bins = {}
        self.mapping = {}
        self.bins_np = np.array([])

    def fit(self, X: DataFrame, y: Series = None) -> "Transformer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'Transformer'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.columns = util.get_numerical_columns(X)
        self.column_names = self.get_column_names(
            inplace=False, columns=self.columns, suffix="bin"
        )
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        if self.idx_columns.size == 0:
            return self
        self.bins_dict, self.pretty_bins_dict, self.bins_np = self.compute_bins(
            X[self.columns], y
        )
        self.labels, self.labels_np = self.get_labels(self.pretty_bins_dict)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        if self.idx_columns.size == 0:
            return X

        new_series_list = []
        for c, n in zip(self.columns, self.column_names):
            n_bins = len(self.bins_dict[c])
            dummy = X[c].where(~(X[c] < self.bins_dict[c][1]), self.labels[c][0])
            for j in range(1, n_bins - 1):
                dummy = dummy.where(
                    ~(
                        (X[c] >= self.bins_dict[c][j])
                        & (X[c] < self.bins_dict[c][j + 1])
                    ),
                    self.labels[c][j],
                )
            dummy = dummy.where(~(X[c] > self.bins_dict[c][-2]), self.labels[c][-1])
            new_series_list.append(dummy.rename(n))

        X_binning = util.get_function(X).concat(new_series_list, axis=1)
        if self.inplace:
            columns_dict = dict(zip(self.column_names, self.columns))
            if len(self.base_columns) == len(self.column_names):
                return X_binning.rename(columns=columns_dict)
            return (
                util.get_function(X)
                .concat([X.drop(self.columns, axis=1), X_binning], axis=1)
                .rename(columns=columns_dict)[self.base_columns]
            )
        return util.get_function(X).concat([X, X_binning], axis=1)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X : np.ndarray
             Array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        X_bin = binning_new(
            X[:, self.idx_columns].astype(float), self.bins_np, self.labels_np
        )
        if self.inplace:
            X = X.astype(object)
            X[:, self.idx_columns] = X_bin
            return X
        return np.concatenate((X, X_bin), axis=1)

    @staticmethod
    def get_labels(pretty_bins_dict: Dict[str, np.array]):
        """Get the labels of the bins.

        Parameters
        ----------
        pretty_bins_dict : Dict[str, np.array])
            pretified bins used to generate the labels.

        Returns
        -------
        Dict[str, np.array]
            Labels.
        np.array
            Labels.
        """
        labels = {}
        for col, bins in pretty_bins_dict.items():
            if len(bins) == 2:
                labels[col] = ["(-inf, inf)"]
            else:
                labels[col] = (
                    [f"(-inf, {bins[1]})"]
                    + [f"[{b1}, {b2})" for b1, b2 in zip(bins[1:-2], bins[2:-1])]
                    + [f"[{bins[-2]}, inf)"]
                )
        labels_np = (
            pd.DataFrame(
                {k: {i: x for i, x in enumerate(v)} for k, v in labels.items()}
            )
            .fillna(0)
            .to_numpy()
        )
        return labels, labels_np
