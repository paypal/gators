# License: Apache-2.0
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


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

    value (Union[float, str, None]) : Imputation value, default None.
        used for `strategy=constant`.
    theta_vec : List[float], default None.
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
        self.statistics_np: np.ndarray = None
        self.idx_columns: np.ndarray = None
        self.X_dtypes: Series = None

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
        return util.get_function(X).fillna(X, value=self.statistics)

    def compute_statistics(
        self, X: DataFrame, value: Union[float, int, str, None]
    ) -> Dict[str, Union[float, int, str]]:
        """Compute the imputation values.

        Parameters
        ----------
        X : DataFrame
            Dataframe. used to compute the imputation values.
        value : Union[float, int, str, None]
            Value used for imputation.

        Returns
        -------
        statistics : Dict[str, Union[float, int, str]]
            Imputation value mapping.
        """
        if self.strategy == "mean":
            statistics = util.get_function(X).to_dict(X[self.columns].mean())
        elif self.strategy == "median":
            statistics = util.get_function(X).to_dict(X[self.columns].median())
        elif self.strategy == "most_frequent":
            statistics = util.get_function(X).most_frequent(X[self.columns])
        else:  # strategy == 'constant'
            values = len(self.columns) * [value]
            statistics = dict(zip(self.columns, values))
        if pd.Series(statistics).isnull().sum():
            raise ValueError(
                """Some columns contains only NaN values and the
                imputation values cannot be calculated.
                Remove these columns
                before performing the imputation
                (e.g. with `gators.data_cleaning.drop_high_nan_ratio()`)."""
            )
        return statistics
