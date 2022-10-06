# Licence Apache-2.0
from math import pi
from typing import List

import numpy as np

import feature_gen_dt

from ._base_datetime_feature import _BaseDatetimeFeature

from gators import DataFrame


class CyclicHourOfDay(_BaseDatetimeFeature):
    """Create new columns based on the cyclic mapping of the hour of the day.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation_dt import CyclicHourOfDay
    >>> obj = CyclicHourOfDay(columns=['A'])

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
                        A  B  A__hour_of_day_cos  A__hour_of_day_sin
    0 2020-01-01 23:00:00  0        9.659258e-01           -0.258819
    1 2020-12-15 18:00:00  1       -1.836970e-16           -1.000000
    2                 NaT  0                 NaN                 NaN

    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18',  None], 'B': [0, 1, 0]})
    >>> X['A'] = X['A'].astype('datetime64[ns]')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 0.9659258262890681,
            -0.2588190451025215],
           [Timestamp('2020-12-15 18:00:00'), 1, -1.8369701987210297e-16,
            -1.0],
           [NaT, 0, nan, nan]], dtype=object)

    """

    def __init__(self, columns: List[str], date_format: str = "ymd"):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = self.get_cyclic_column_names(columns, "hour_of_day")
        self.prefactor = 2 * pi / 24.0
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
        return self.compute_cyclic_hour_of_day(X, self.prefactor)

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
        X_new = feature_gen_dt.cyclic_hour_of_day(
            X[:, self.idx_columns], self.prefactor
        )
        return np.concatenate([X, X_new], axis=1)

    def compute_cyclic_hour_of_day(self, X: DataFrame, prefactor: float) -> DataFrame:
        """Compute the cyclic hours of the day features.

        Parameters
        ----------
        X_datetime : DataFrame
            Dataframe of datetime columns.
        prefactor : float
            Prefactor.

        Returns
        -------
        X : DataFrame
            Dataframe of cyclic hours of the day features.
        """

        for i, c in enumerate(self.columns):
            hour = X[c].dt.hour
            X[self.column_names[2 * i]] = np.cos(prefactor * hour)
            X[self.column_names[2 * i + 1]] = np.sin(prefactor * hour)
        return X
