# Licence Apache-2.0
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers.transformer import Transformer
from ..util import util


class _BaseDatetimeFeature(Transformer):
    """Base datetime transformer class.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    column_names : List[str], default to None.
        List of column names.
    column_mapping: Dict[str, List[str]]
        Mapping between generated features and base features.

    """

    def __init__(
        self,
        columns: List[str],
        column_names: List[str],
        column_mapping: Dict[str, str],
    ):
        Transformer.__init__(self)
        self.columns = columns
        self.column_names = column_names
        self.column_mapping = column_mapping
        self.idx_columns: np.ndarray = np.array([])
        self.n_columns = len(self.columns)

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "_BaseDatetimeFeature":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Target values.

        Returns
        -------
        _BaseDatetimeFeature
            Instance of itself.
        """
        self.check_dataframe(X)
        X_datetime_dtype = X.iloc[:100][self.columns].dtypes
        for column in self.columns:
            if not np.issubdtype(X_datetime_dtype[column], np.datetime64):
                raise TypeError(
                    """
                    Datetime columns should be of subtype np.datetime64.
                    Use `ConvertColumnDatatype` to convert the dtype.
                """
                )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns,
            selected_columns=self.columns,
        )
        return self

    @staticmethod
    def get_cyclic_column_names(columns: List[str], pattern: str):
        """Get the column names.

        Parameters
        ----------
        columns : List[str]
            List of datetime features.
        pattern: str
            Pattern.
        """
        column_names = []
        for c in columns:
            column_names.append(f"{c}__{pattern}_cos")
            column_names.append(f"{c}__{pattern}_sin")
        return column_names
