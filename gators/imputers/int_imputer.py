# License: Apache-2.0
import warnings
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from imputer import float_imputer, float_imputer_object

from ..util import util
from ._base_imputer import _BaseImputer


class IntImputer(_BaseImputer):
    """Impute the numerical columns satisfying the condition X == X.round()
    using the strategy passed by the user.

    Parameters
    ----------
    strategy : str
        Imputation strategy.

        Supported imputation strategies are:

            - 'constant'
            - 'most_frequent'
            - 'mean'
            - 'median'

    value : str, default to None.
        Imputation value used for `strategy=constant`.

    columns: List[str], default to None.
        List of columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gators.imputers import IntImputer
    >>> X = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = IntImputer(strategy='constant', value=-999)
    >>> obj.fit_transform(X)
           A  B
    0    1.0  z
    1    2.0  a
    2 -999.0  a

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.imputers import IntImputer
    >>> X = ks.DataFrame({'A': [1, 2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = IntImputer(strategy='constant', value=-999)
    >>> obj.fit_transform(X)
           A  B
    0    1.0  z
    1    2.0  a
    2 -999.0  a

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gators.imputers import IntImputer
    >>> X = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = IntImputer(strategy='constant', value=-999)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.0, 'z'],
           [2.0, 'a'],
           [-999.0, 'a']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.imputers import IntImputer
    >>> X = ks.DataFrame({'A': [1, 2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = IntImputer(strategy='constant', value=-999)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.0, 'z'],
           [2.0, 'a'],
           [-999.0, 'a']], dtype=object)

    See Also
    --------
    gators.imputers.FloatImputer
        Impute float columns.
    gators.imputers.NumericsImputer
        Impute numerical columns.
    gators.imputers.ObjectImputer
        Impute categorical columns.

    """

    def __init__(self, strategy: str, value: float = None, columns: List[str] = None):
        _BaseImputer.__init__(self, strategy, value, columns)
        if strategy == "constant" and not isinstance(value, int):
            raise TypeError(
                """`value` should be a integer
                for the IntImputer class"""
            )
        self.columns = columns

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "IntImputer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
            Imputer: Instance of itself.
        """
        self.check_dataframe(X)
        if not self.columns:
            self.columns = util.get_int_only_columns(X=X)
        if not self.columns:
            warnings.warn(
                """`X` does not contain columns satisfying:
                X[column] == X[column].round(),
                `IntImputer` is not needed"""
            )
            self.idx_columns = np.array([])
            return self
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        self.statistics = self.compute_statistics(
            X=X,
            columns=self.columns,
            strategy=self.strategy,
            value=self.value,
        )
        self.statistics_values = np.array(list(self.statistics.values())).astype(
            np.float64
        )
        return self

    def transform_numpy(self, X: Union[pd.Series, ks.Series], y=None):
        """Transform the numpy ndarray X.

        Parameters
        ----------
        X (np.ndarray): Input ndarray.

        Returns
        -------
            np.ndarray: Imputed ndarray.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        X_dtype = X.dtype
        if "int" in str(X_dtype):
            return X
        elif X_dtype == object:
            return float_imputer_object(
                X, self.statistics_values.astype(object), self.idx_columns
            )
        else:
            return float_imputer(X, self.statistics_values, self.idx_columns)
