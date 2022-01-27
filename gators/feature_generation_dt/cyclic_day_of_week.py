# Licence Apache-2.0
from math import pi
from typing import List

import numpy as np

import feature_gen_dt

from ._base_datetime_feature import _BaseDatetimeFeature

PREFACTOR = 2 * pi / 6.0


from gators import DataFrame


class CyclicDayOfWeek(_BaseDatetimeFeature):
    """Create new columns based on the cyclic mapping of the day of the week.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_dt import CyclicDayOfWeek
    >>> obj = CyclicDayOfWeek(columns=['A'])

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
                        A  B  A__day_of_week_cos  A__day_of_week_sin
    0 2020-01-01 23:00:00  0                -0.5            0.866025
    1 2020-12-15 18:00:00  1                 0.5            0.866025
    2                 NaT  0                 NaN                 NaN


    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18',  None], 'B': [0, 1, 0]})
    >>> X['A'] = X['A'].astype('datetime64[ns]')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, -0.4999999999999998,
            0.8660254037844388],
           [Timestamp('2020-12-15 18:00:00'), 1, 0.5000000000000001,
            0.8660254037844386],
           [NaT, 0, nan, nan]], dtype=object)

    """

    def __init__(self, columns: List[str], date_format: str = "ymd"):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = self.get_cyclic_column_names(columns, "day_of_week")
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
        return self.compute_cyclic_day_of_week(X)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        X_new = feature_gen_dt.cyclic_day_of_week(X[:, self.idx_columns], PREFACTOR)
        return np.concatenate([X, X_new], axis=1)

    def compute_cyclic_day_of_week(self, X: DataFrame) -> DataFrame:
        """Compute the cyclic day of the week features.

        Parameters
        ----------
        X : DataFrame
            Dataframe. of datetime columns.

        Returns
        -------
        X : DataFrame
            Dataframe of cyclic day of the week features.
        """

        for i, c in enumerate(self.columns):
            dayofweek = X[c].dt.dayofweek
            X[self.column_names[2 * i]] = np.cos(PREFACTOR * dayofweek)
            X[self.column_names[2 * i + 1]] = np.sin(PREFACTOR * dayofweek)
        return X
