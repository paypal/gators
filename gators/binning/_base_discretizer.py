# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from pyspark.ml.feature import Bucketizer

from binning import discretizer, discretizer_inplace

from ..transformers.transformer import Transformer
from ..util import util

EPSILON = 1e-10


class _BaseDiscretizer(Transformer):
    """Base discretizer transformer class.

    Parameters
    ----------
    n_bins : int
        Number of bins to use.
    inplace : bool
        If False, return the dataframe with the new discretized columns
        with the names '`column_name`__bin'). Otherwise,
        return the dataframe with the existing binned columns.

    """

    def __init__(self, n_bins: int, inplace: bool):
        if not isinstance(n_bins, int):
            raise TypeError("`n_bins` should be an int.")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        Transformer.__init__(self)
        self.n_bins = n_bins
        self.inplace = inplace
        self.columns = []
        self.output_columns = []
        self.idx_columns: np.ndarray = np.array([])
        self.bins = {}
        self.bins_np = np.array([])
        self.bins_ks: List[List[float]] = [[]]

    def fit(self, X: Union[pd.DataFrame, ks.DataFrame], y=None) -> "_BaseDiscretizer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        'Discretizer'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_numerical_columns(X)
        self.output_columns = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        if self.idx_columns.size == 0:
            return self

        self.bins, self.bins_np = self.compute_bins(X[self.columns], self.n_bins)
        return self

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame]
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Transformed dataframe.
        """
        self.check_dataframe(X)
        if self.idx_columns.size == 0:
            return X
        if isinstance(X, pd.DataFrame):
            if self.inplace:
                return self.bin_pd_inplace(
                    X, self.columns, self.output_columns, self.bins
                )
            return self.bin_pd(X, self.columns, self.output_columns, self.bins)
        if self.inplace:
            return self.bin_ks_inplace(X, self.columns, self.output_columns, self.bins)[
                X.columns
            ]

        return self.bin_ks(X, self.columns, self.output_columns, self.bins)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array.

        Parameters
        ----------
        X : np.ndarray
            NumPy array.

        Returns
        -------
        np.ndarray
            Transformed NumPy array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        if self.inplace:
            if X.dtype == object:
                return discretizer_inplace(X, self.bins_np, self.idx_columns)
            return discretizer_inplace(X.astype(object), self.bins_np, self.idx_columns)
        if X.dtype == object:
            return discretizer(X, self.bins_np, self.idx_columns)
        return discretizer(X.astype(object), self.bins_np, self.idx_columns)

    @staticmethod
    def bin_pd_inplace(
        X: pd.DataFrame, columns: List[str], output_columns: List[str], bins
    ):
        """Perform the binning inplace for pandas dataframes.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe.
        columns : List[str]
            Columns to be binnned.
        output_columns : List[str]
            Binnned column names.
        bins : [type]
            [description]

        Returns
        -------
        pd.DataFrame
            Dataframe.
        """

        def f(x, bins, columns):
            name = x.name
            if name not in columns:
                return x
            return (
                pd.cut(
                    x,
                    bins[name],
                    labels=np.arange(len(bins[name]) - 1),
                    duplicates="drop",
                )
                .fillna(0)
                .astype(float)
                .astype(str)
            )

        return X.apply(f, args=(bins, columns))

    @staticmethod
    def bin_pd(X, columns, output_columns, bins):
        """Perform the binning not inplace for pandas dataframes.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe.
        columns : List[str]
            Columns to be binnned.
        output_columns : List[str]
            Binnned column names.
        bins : [type]
            [description]

        Returns
        -------
        pd.DataFrame
            Dataframe.
        """

        def f(x, bins, columns):
            name = x.name
            return (
                pd.cut(
                    x,
                    bins[name],
                    labels=np.arange(len(bins[name]) - 1),
                    duplicates="drop",
                )
                .fillna(0)
                .astype(float)
                .astype(str)
            )

        return X.join(
            X[columns]
            .apply(f, args=(bins, columns))
            .astype(object)
            .rename(columns=dict(zip(columns, output_columns)))
        )

    @staticmethod
    def bin_ks_inplace(X, columns, output_columns, bins):
        """Perform the binning not inplace for kolas dataframes.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe.
        columns : List[str]
            Columns to be binnned.
        output_columns : List[str]
            Binnned column names.
        bins : [type]
            [description]

        Returns
        -------
        ks.DataFrame
            Dataframe.
        """
        X = (
            Bucketizer(splitsArray=bins, inputCols=columns, outputCols=output_columns)
            .transform(X.to_spark())
            .to_koalas()
            .drop(columns, axis=1)
            .rename(columns=dict(zip(output_columns, columns)))
        )
        X[columns] = X[columns].astype(str)
        return X

    @staticmethod
    def bin_ks(X, columns, output_columns, bins):
        """Perform the binning inplace for kolas dataframes.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe.
        columns : List[str]
            Columns to be binnned.
        output_columns : List[str]
            Binnned column names.
        bins : [type]
            [description]

        Returns
        -------
        ks.DataFrame
            Dataframe.
        """
        X = (
            Bucketizer(splitsArray=bins, inputCols=columns, outputCols=output_columns)
            .transform(X.to_spark())
            .to_koalas()
        )
        X[output_columns] = X[output_columns].astype(str)
        return X
