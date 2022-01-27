# Licence Apache-2.0
from typing import List

import numpy as np
import pandas as pd

import feature_gen_dt

from ._base_datetime_feature import _BaseDatetimeFeature

from gators import DataFrame


class OrdinalDayOfWeek(_BaseDatetimeFeature):
    """Create new columns based on the day of the week.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_dt import OrdinalDayOfWeek
    >>> obj = OrdinalDayOfWeek(columns=['A'])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18',  None], 'B': [0, 1, 0]}), npartitions=1)
    >>> X['A'] = X['A'].astype('datetime64[ns]')

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18',  None], 'B': [0, 1, 0]})
    >>> X['A'] = X['A'].astype('datetime64[ns]')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18',  None], 'B': [0, 1, 0]})
    >>> X['A'] = X['A'].astype('datetime64[ns]')

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
                        A  B  A__day_of_week
    0 2020-01-01 23:00:00  0             2.0
    1 2020-12-15 18:00:00  1             1.0
    2                 NaT  0             NaN

    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18',  None], 'B': [0, 1, 0]})
    >>> X['A'] = X['A'].astype('datetime64[ns]')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 2.0],
           [Timestamp('2020-12-15 18:00:00'), 1, 1.0],
           [NaT, 0, nan]], dtype=object)

    """

    def __init__(self, columns: List[str], date_format: str = "ymd"):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = [f"{c}__day_of_week" for c in columns]
        _BaseDatetimeFeature.__init__(self, columns, date_format, column_names)

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

        for name, col in zip(self.column_names, self.columns):
            X[name] = X[col].dt.dayofweek

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
        X_new = feature_gen_dt.ordinal_day_of_week(X[:, self.idx_columns])
        return np.concatenate([X, X_new], axis=1)
