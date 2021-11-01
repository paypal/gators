# License: Apache-2.0
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer


class _BaseImputer(Transformer):
    """Base imputer transformer class.

    Parameters
    ----------
    strategy : str
        Imputation strategy. The possible values are:

        * constant
        * most_frequent (only for the FloatImputer class)
        * mean (only for the FloatImputer class)
        * median (only for the FloatImputer class)

    value (Union[float, str, None]): Imputation value, default to None.
        used for `strategy=constant`.
    columns: List[str], default to None.
        List of columns.

    """

    def __init__(
        self, strategy: str, value: Union[float, str, None], columns: List[str]
    ):
        if not isinstance(strategy, str):
            raise TypeError("`strategy` should be a string.")
        if strategy == "constant" and value is None:
            raise ValueError('if `strategy` is "constant", `value` should not be None.')
        if strategy not in ["constant", "mean", "median", "most_frequent"]:
            raise ValueError("Imputation `strategy` not implemented.")
        if not isinstance(columns, list) and columns is not None:
            raise TypeError("`columns` should be a list or None.")

        Transformer.__init__(self)
        self.strategy = strategy
        self.value = value
        self.columns = columns
        self.statistics: Dict = {}
        self.statistics_values: np.ndarray = None
        self.idx_columns: np.ndarray = None
        self.X_dtypes: Union[pd.Series, ks.Series] = None

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
        if isinstance(X, pd.DataFrame):
            return X.fillna(self.statistics)
        for col, val in self.statistics.items():
            X[col] = X[col].fillna(val)
        return X

    @staticmethod
    def compute_statistics(
        X: Union[pd.DataFrame, ks.DataFrame],
        columns: List[str],
        strategy: str,
        value: Union[float, int, str, None],
    ) -> Dict[str, Union[float, int, str]]:
        """Compute the imputation values.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Dataframe used to compute the imputation values.
        columns : List[str]
            Columns to consider.
        strategy : str
            Imputation strategy.
        value : Union[float, int, str, None]
            Value used for imputation.

        Returns
        -------
        Dict[str, Union[float, int, str]]
            Imputation value mapping.
        """
        if strategy == "mean":
            statistics = X[columns].astype(np.float64).mean().to_dict()
        elif strategy == "median":
            statistics = X[columns].astype(np.float64).median().to_dict()
        elif strategy == "most_frequent":
            values = [X[c].value_counts().index.to_numpy()[0] for c in columns]
            statistics = dict(zip(columns, values))
        else:  # strategy == 'constant'
            values = len(columns) * [value]
            statistics = dict(zip(columns, values))
        if pd.Series(statistics).isnull().sum():
            raise ValueError(
                """Some columns contains only NaN values and the
                imputation values cannot be calculated.
                Remove these columns
                before performing the imputation
                (e.g. with `gators.data_cleaning.drop_high_nan_ratio()`)."""
            )
        return statistics
