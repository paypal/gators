# Licence Apache-2.0
from typing import List

import numpy as np

import feature_gen_dt

from ..transformers import Transformer
from ..util import util

from gators import DataFrame, Series


class DeltaTime(Transformer):
    """Create new columns based on the time difference in sec. between two columns.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_dt import DeltaTime
    >>> obj = DeltaTime(columns_a=['C'], columns_b=['A'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(
    ... pd.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-02T00',  None],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-15T23', '2020-01-03T05',  None]}), npartitions=1)
    >>> X[['A', 'C']] = X[['A', 'C']].astype('datetime64[ns]')

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-02T00',  None],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-15T23', '2020-01-03T05',  None]})
    >>> X[['A', 'C']] = X[['A', 'C']].astype('datetime64[ns]')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-02T00',  None],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-15T23', '2020-01-03T05',  None]})
    >>> X[['A', 'C']] = X[['A', 'C']].astype('datetime64[ns]')

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
                        A  B                   C  C__A__Deltatime[s]
    0 2020-01-01 23:00:00  0 2020-01-15 23:00:00           1209600.0
    1 2020-01-02 00:00:00  1 2020-01-03 05:00:00            104400.0
    2                 NaT  0                 NaT                 NaN

    >>> X = pd.DataFrame({
    ... 'A': ['2020-01-01T23', '2020-01-02T00',  None],
    ... 'B': [0, 1, 0],
    ... 'C': ['2020-01-15T23', '2020-01-03T05',  None]})
    >>> X[['A', 'C']] = X[['A', 'C']].astype('datetime64[ns]')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0,
            Timestamp('2020-01-15 23:00:00'), 1209600.0],
           [Timestamp('2020-01-02 00:00:00'), 1,
            Timestamp('2020-01-03 05:00:00'), 104400.0],
           [NaT, 0, NaT, nan]], dtype=object)
    """

    def __init__(self, columns_a: List[str], columns_b: List[str]):
        Transformer.__init__(self)
        if not isinstance(columns_a, (list, np.ndarray)):
            raise TypeError("`columns_a` should be a list.")
        if not columns_a:
            raise ValueError("`columns_a` should not be empty.")
        if not isinstance(columns_b, (list, np.ndarray)):
            raise TypeError("`columns_b` should be a list.")
        if not columns_b:
            raise ValueError("`columns_b` should not be empty.")
        if len(columns_b) != len(columns_a):
            raise ValueError("`columns_a` and `columns_b` should have the same length.")
        self.unit = "s"
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.deltatime_dtype = f"timedelta64[{self.unit}]"
        self.column_names = [
            f"{c_a}__{c_b}__Deltatime[{self.unit}]"
            for c_a, c_b in zip(columns_a, columns_b)
        ]

    def fit(self, X: DataFrame, y: Series = None) -> "DeltaTime":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : DeltaTime
            Instance of itself.
        """
        self.check_dataframe(X)
        columns = list(set(self.columns_a + self.columns_b))
        columns = [c for c in X.columns if c in columns]
        X_datetime_dtype = X.dtypes
        for column in columns:
            if not np.issubdtype(X_datetime_dtype[column], np.datetime64):
                raise TypeError(
                    """
                    Datetime columns should be of subtype np.datetime64.
                    Use `ConvertColumnDatatype` to convert the dtype.
                """
                )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns,
            selected_columns=columns,
        )
        self.idx_columns_a = util.get_idx_columns(
            # columns=X.columns,
            columns=columns,
            selected_columns=self.columns_a,
        )
        self.idx_columns_b = util.get_idx_columns(
            # columns=X.columns,
            columns=columns,
            selected_columns=self.columns_b,
        )
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
        X = util.get_function(X).delta_time(
            X, self.column_names, self.columns_a, self.columns_b, self.deltatime_dtype
        )
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray:
            Transformed array.
        """
        self.check_array(X)
        X_new = feature_gen_dt.deltatime(
            X[:, self.idx_columns], self.idx_columns_a, self.idx_columns_b
        )
        return np.concatenate([X, X_new], axis=1)
