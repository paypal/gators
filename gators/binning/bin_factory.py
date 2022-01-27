# License: Apache-2.0
from abc import ABC
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from gators import DataFrame


EPSILON = 1e-10


class BinFactory(ABC):
    def bin(self) -> DataFrame:
        """Add the binned columns to the input dataframe."""

    def bin_inplace(self) -> DataFrame:
        """Add in-place the binned columns to the input dataframe."""


class BinPandas(BinFactory):
    def bin(
        self,
        X: DataFrame,
        splits: pd.DataFrame,
        labels: Dict[str, str],
        columns: List[str],
        output_columns: List[str],
    ) -> DataFrame:
        """Add the binned columns to the input Pandas dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe
        splits : pd.DataFrame
            Dictionary of splits. The keys are the column names, the values are the split arrays.
        labels : Dict[str, str]
            Dictionary of labels. The keys are the column names, the values are the label arrays.
        columns : List[str]
            List of columns.
        output_columns : List[str]
             List of output columns, only used where `inplace=True`.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """

        for name, col in zip(output_columns, columns):
            X[name] = pd.cut(
                X[col],
                bins=splits[col],
                labels=labels[col],
                duplicates="drop",
            ).astype(object)
        return X

    def bin_inplace(
        self,
        X: DataFrame,
        splits: pd.DataFrame,
        labels: Dict[str, str],
        columns: List[str],
        output_columns: List[str],
    ) -> DataFrame:
        """Add in-place the binned columns to the input Pandas dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe
        splits : pd.DataFrame
            Dictionary of splits. The keys are the column names, the values are the split arrays.
        labels : Dict[str, str]
            Dictionary of labels. The keys are the column names, the values are the label arrays.
        columns : List[str]
            List of columns.
        output_columns : List[str]
             List of output columns, only used where `inplace=True`.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        for name, col in zip(output_columns, columns):
            X[col] = pd.cut(
                X[col],
                bins=splits[col],
                labels=labels[col],
                duplicates="drop",
            ).astype(object)
        return X


class BinKoalas(BinFactory):
    def bin(
        self,
        X: DataFrame,
        splits: pd.DataFrame,
        labels: Dict[str, str],
        columns: List[str],
        output_columns: List[str],
    ) -> DataFrame:
        """Add the binned columns to the input Koalas dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe
        splits : pd.DataFrame
            Dictionary of splits. The keys are the column names, the values are the split arrays.
        labels : Dict[str, str]
            Dictionary of labels. The keys are the column names, the values are the label arrays.
        columns : List[str]
            List of columns.
        output_columns : List[str]
             List of output columns, only used where `inplace=True`.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        from pyspark.ml.feature import Bucketizer

        bins_np = [np.unique(b) + EPSILON for b in splits.values()]
        X = (
            Bucketizer(
                splitsArray=bins_np, inputCols=columns, outputCols=output_columns
            )
            .transform(X.to_spark())
            .to_koalas()
        )
        X[output_columns] = "_" + X[output_columns].astype(int).astype(str)
        return X

    def bin_inplace(
        self,
        X: DataFrame,
        splits: pd.DataFrame,
        labels: Dict[str, str],
        columns: List[str],
        output_columns: List[str],
    ) -> DataFrame:
        """Add in-place the binned columns to the input Koalas dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe
        splits : pd.DataFrame
            Dictionary of splits. The keys are the column names, the values are the split arrays.
        labels : Dict[str, str]
            Dictionary of labels. The keys are the column names, the values are the label arrays.
        columns : List[str]
            List of columns.
        output_columns : List[str]
             List of output columns, only used where `inplace=True`.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        from pyspark.ml.feature import Bucketizer

        bins_np = [np.unique(b) + EPSILON for b in splits.values()]
        ordered_columns = X.columns
        X = (
            Bucketizer(
                splitsArray=bins_np, inputCols=columns, outputCols=output_columns
            )
            .transform(X.to_spark())
            .to_koalas()
            .drop(columns, axis=1)
            .rename(columns=dict(zip(output_columns, columns)))
        )
        X[columns] = "_" + X[columns].astype(int).astype(str)
        return X[ordered_columns]


class BinDask(BinFactory):
    def bin(
        self,
        X: DataFrame,
        splits: pd.DataFrame,
        labels: Dict[str, str],
        columns: List[str],
        output_columns: List[str],
    ) -> DataFrame:
        """Add the binned columns to the input Dask dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe
        splits : pd.DataFrame
            Dictionary of splits. The keys are the column names, the values are the split arrays.
        labels : Dict[str, str]
            Dictionary of labels. The keys are the column names, the values are the label arrays.
        columns : List[str]
            List of columns.
        output_columns : List[str]
             List of output columns, only used where `inplace=True`.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        for name, col in zip(output_columns, columns):
            X[name] = (
                X[col]
                .map_partitions(
                    pd.cut,
                    bins=splits[col],
                    labels=labels[col],
                    duplicates="drop",
                )
                .astype(object)
            )
        return X

    def bin_inplace(
        self,
        X: DataFrame,
        splits: pd.DataFrame,
        labels: Dict[str, str],
        columns: List[str],
        output_columns: List[str],
    ) -> DataFrame:
        """Add in-place the binned columns to the input Dask dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe
        splits : pd.DataFrame
            Dictionary of splits. The keys are the column names, the values are the split arrays.
        labels : Dict[str, str]
            Dictionary of labels. The keys are the column names, the values are the label arrays.
        columns : List[str]
            List of columns.
        output_columns : List[str]
             List of output columns, only used where `inplace=True`.

        Returns
        -------
        DataFrame
            Transformed dataframe.
        """
        for name, col in zip(output_columns, columns):
            X[col] = (
                X[col]
                .map_partitions(
                    pd.cut,
                    bins=splits[col],
                    labels=labels[col],
                    duplicates="drop",
                )
                .astype(object)
            )
        return X


def get_bin(X: DataFrame) -> Union[BinPandas, BinKoalas, BinDask]:
    """Return the `Bin` class based on the dataframe library.

    Parameters
    ----------
    X : DataFrame
        Dataframe.

    Returns
    -------
    Union[BinPandas, BinKoalas, BinDask]
        `Bin` class assocaited to the dataframe librarry.
    """
    factories = {
        "<class 'pandas.core.frame.DataFrame'>": BinPandas(),
        "<class 'databricks.koalas.frame.DataFrame'>": BinKoalas(),
        "<class 'dask.dataframe.core.DataFrame'>": BinDask(),
    }
    return factories[str(type(X))]
