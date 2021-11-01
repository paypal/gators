# Licence Apache-2.0
from math import pi
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

import feature_gen_dt

from ._base_datetime_feature import _BaseDatetimeFeature

TWO_PI = 2 * pi


class CyclicDayOfMonth(_BaseDatetimeFeature):
    """Create new columns based on the cyclic mapping of the day of the month.

    Parameters
    ----------
    columns: List[str]
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__day_of_month_cos  A__day_of_month_sin
    0 2020-01-01 23:00:00  0             1.000000             0.000000
    1 2020-12-15 18:00:00  1            -0.978148             0.207912
    2                 NaT  0                  NaN                  NaN

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B  A__day_of_month_cos  A__day_of_month_sin
    0 2020-01-01 23:00:00  0             1.000000             0.000000
    1 2020-12-15 18:00:00  1            -0.978148             0.207912
    2                 NaT  0                  NaN                  NaN

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = pd.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 1.0, 0.0],
           [Timestamp('2020-12-15 18:00:00'), 1, -0.9781476007338057,
            0.2079116908177593],
           [NaT, 0, nan, nan]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import CyclicDayOfMonth
    >>> X = ks.DataFrame({'A': ['2020-01-01T23', '2020-12-15T18', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = CyclicDayOfMonth(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, 1.0, 0.0],
           [Timestamp('2020-12-15 18:00:00'), 1, -0.9781476007338057,
            0.2079116908177593],
           [NaT, 0, nan, nan]], dtype=object)


    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = self.get_cyclic_column_names(columns, "day_of_month")
        column_mapping = {
            name: col for name, col in zip(column_names, columns + columns)
        }
        _BaseDatetimeFeature.__init__(self, columns, column_names, column_mapping)

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
        return self.compute_cyclic_day_of_month(X, self.columns, self.column_names)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array `X`.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Transformed array.
        """

        self.check_array(X)
        return feature_gen_dt.cyclic_day_of_month(X, self.idx_columns)

    @staticmethod
    def compute_cyclic_day_of_month(
        X: Union[pd.DataFrame, ks.DataFrame],
        columns: List[str],
        column_names: List[str],
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Compute the cyclic day of the month features.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Dataframe of datetime columns.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Dataframe of cyclic day of the month features.
        """

        def f_cos(x):
            day_of_month = x.dt.day - 1
            n_days_in_month = x.dt.daysinmonth - 1
            prefactors = 2 * np.pi / n_days_in_month
            return np.cos(prefactors * day_of_month)

        def f_sin(x):
            day_of_month = x.dt.day - 1
            n_days_in_month = x.dt.daysinmonth - 1
            prefactors = 2 * np.pi / n_days_in_month
            return np.sin(prefactors * day_of_month)

        if isinstance(X, pd.DataFrame):
            for i, col in enumerate(columns):
                X_cos = X[[col]].apply(f_cos)
                X_cos.columns = [column_names[2 * i]]
                X_sin = X[[col]].apply(f_sin)
                X_sin.columns = [column_names[2 * i + 1]]
                X = X.join(X_cos.join(X_sin))
            return X

        for i, col in enumerate(columns):
            n_days_in_month = X[col].dt.daysinmonth - 1
            prefactors = 2 * np.pi / n_days_in_month
            X = X.assign(
                dummy_cos=np.cos(prefactors * (X[col].dt.day - 1.0)),
                dummy_sin=np.sin(prefactors * (X[col].dt.day - 1.0)),
            ).rename(
                columns={
                    "dummy_cos": column_names[2 * i],
                    "dummy_sin": column_names[2 * i + 1],
                }
            )
        return X
