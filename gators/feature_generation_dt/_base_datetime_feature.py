# Licence Apache-2.0

from typing import List, Tuple

import numpy as np

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class _BaseDatetimeFeature(Transformer):
    """Base datetime transformer class.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    column_names : List[str], default None.
        List of column names.
    """

    def __init__(
        self,
        columns: List[str],
        date_format: str,
        column_names: List[str],
    ):
        if not isinstance(date_format, str):
            raise TypeError("`date_format` should be a string.")
        if sorted(list(date_format)) != ["d", "m", "y"]:
            raise ValueError(
                "`date_format` should be a string composed of the letters `d`, `m` and `y`."
            )
        Transformer.__init__(self)
        self.columns = columns
        self.column_names = column_names
        self.date_format = date_format

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
        self : Transformer
            Instance of itself.
        """
        self.check_dataframe(X)
        X_datetime_dtype = X[self.columns].dtypes
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
        self.n_columns = len(self.columns)
        self.idx_day_bounds, self.idx_month_bounds, self.idx_year_bounds = self.get_idx(
            self.date_format
        )
        return self

    @staticmethod
    def get_cyclic_column_names(columns: List[str], pattern: str):
        """Get the column names.

        Parameters
        ----------
        theta_vec : List[float]
            List of datetime features.
        pattern: str
            Pattern.
        """
        column_names = []
        for c in columns:
            column_names.append(f"{c}__{pattern}_cos")
            column_names.append(f"{c}__{pattern}_sin")
        return column_names

    @staticmethod
    def get_idx(date_format: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """[summary]

        Parameters
        ----------
        date_format : str
            Datetime format

        Returns
        -------
        idx_day_bounds : np.ndarray
            Start and end indices of the day.

        idx_month : np.ndarray
            Start and end indices of the month.

        idx_year_bounds : np.ndarray
            Start and end indices of the year.
        """
        idx_start_day = 3 * date_format.index("d")
        idx_start_month = 3 * date_format.index("m")
        idx_start_year = 3 * date_format.index("y")
        idx_start_day = (
            idx_start_day if idx_start_year > idx_start_day else idx_start_day + 2
        )
        idx_start_month = (
            idx_start_month if idx_start_year > idx_start_month else idx_start_month + 2
        )

        idx_day_bounds = np.array([idx_start_day, idx_start_day + 2])
        idx_month = np.array([idx_start_month, idx_start_month + 2])
        idx_year_bounds = np.array([idx_start_year, idx_start_year + 4])
        return idx_day_bounds, idx_month, idx_year_bounds
