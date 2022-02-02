# License: Apache-2.0
import numpy as np

from binning import binning, binning_inplace

from ..transformers.transformer import Transformer
from ..util import util
from .bin_factory import get_bin

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
        self.columns = []
        self.output_columns = []
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
        self.columns = util.get_numerical_columns(X)
        self.output_columns = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        if self.idx_columns.size == 0:
            return self

        self.bins, self.bins_np = self.compute_bins(X[self.columns], y)
        self.mapping = self.compute_mapping(self.bins)
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
        bin = get_bin(X)
        self.check_dataframe(X)
        if self.idx_columns.size == 0:
            return X
        if self.inplace:
            return bin.bin_inplace(
                X, self.bins, self.mapping, self.columns, self.output_columns
            )
        return bin.bin(X, self.bins, self.mapping, self.columns, self.output_columns)

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
        if self.inplace:
            if X.dtype == object:
                return binning_inplace(X, self.bins_np, self.idx_columns)
            return binning_inplace(X.astype(object), self.bins_np, self.idx_columns)
        if X.dtype == object:
            return binning(X, self.bins_np, self.idx_columns)
        return binning(X.astype(object), self.bins_np, self.idx_columns)

    @staticmethod
    def compute_mapping(bins):
        mapping = {}
        for col in bins.keys():
            if len(bins[col]) == 2:
                mapping[col] = {"_0": "(-inf, inf)"}
            else:
                mapping[col] = dict(
                    zip(
                        [f"_{b}" for b in range(len(bins[col]))],
                        (
                            [f"(-inf, {bins[col][1]}]"]
                            + [
                                f"({b1}, {b2}]"
                                for b1, b2 in zip(bins[col][1:-2], bins[col][2:-1])
                            ]
                            + [f"({bins[col][-2]}, inf)"]
                        ),
                    )
                )
        return mapping
