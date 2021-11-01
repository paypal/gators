# Licence Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

import feature_gen_dt

from ._base_datetime_feature import _BaseDatetimeFeature


class OrdinalMinuteOfHour(_BaseDatetimeFeature):
    """Create new columns based on the minute of the hour.

    Parameters
    ----------
    columns : List[str]
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import OrdinalMinuteOfHour
    >>> X = ks.DataFrame(
    ... {'A': ['2020-01-01T23:00:00', '2020-12-15T18:59:00', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMinuteOfHour(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B A__minute_of_hour
    0 2020-01-01 23:00:00  0               0.0
    1 2020-12-15 18:59:00  1              59.0
    2                 NaT  0               nan

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import OrdinalMinuteOfHour
    >>> X = ks.DataFrame(
    ... {'A': ['2020-01-01T23:00:00', '2020-12-15T18:59:00', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMinuteOfHour(columns=['A'])
    >>> obj.fit_transform(X)
                        A  B A__minute_of_hour
    0 2020-01-01 23:00:00  0               0.0
    1 2020-12-15 18:59:00  1              59.0
    2                 NaT  0               nan

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation_dt import OrdinalMinuteOfHour
    >>> X = ks.DataFrame(
    ... {'A': ['2020-01-01T23:00:00', '2020-12-15T18:59:00', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMinuteOfHour(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, '0.0'],
           [Timestamp('2020-12-15 18:59:00'), 1, '59.0'],
           [NaT, 0, 'nan']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation_dt import OrdinalMinuteOfHour
    >>> X = ks.DataFrame(
    ... {'A': ['2020-01-01T23:00:00', '2020-12-15T18:59:00', pd.NaT], 'B': [0, 1, 0]})
    >>> obj = OrdinalMinuteOfHour(columns=['A'])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[Timestamp('2020-01-01 23:00:00'), 0, '0.0'],
           [Timestamp('2020-12-15 18:59:00'), 1, '59.0'],
           [NaT, 0, 'nan']], dtype=object)


    """

    def __init__(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        column_names = [f"{c}__minute_of_hour" for c in columns]
        column_mapping = dict(zip(column_names, columns))
        _BaseDatetimeFeature.__init__(self, columns, column_names, column_mapping)

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
        if isinstance(X, pd.DataFrame):
            X_ordinal = X[self.columns].apply(
                lambda x: x.dt.minute.astype(np.float64).astype(str)
            )
            X_ordinal.columns = self.column_names
            return X.join(X_ordinal)

        for col, name in zip(self.columns, self.column_names):
            X = X.assign(dummy=X[col].dt.minute.astype(np.float64).astype(str)).rename(
                columns={"dummy": name}
            )
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array X.

        Parameters
        ----------
        X : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray:
            Array with the datetime features added.
        """
        self.check_array(X)
        return feature_gen_dt.ordinal_minute_of_hour(X, self.idx_columns)
