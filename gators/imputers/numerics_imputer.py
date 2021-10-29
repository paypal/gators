# License: Apache-2.0
import warnings
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from imputer import float_imputer, float_imputer_object

from ..util import util
from ._base_imputer import _BaseImputer


class NumericsImputer(_BaseImputer):
    """Impute the numerical columns using the strategy passed by the user.

    Parameters
    ----------
    strategy : str
        Imputation strategy.

        Supported imputation strategies are:

            - 'constant'
            - 'mean'
            - 'median'

    value : str, default to None.
        Imputation value used for `strategy=constant`.

    Examples
    ---------

    * fit & transform with `pandas`

        - impute all the numerical columns

            >>> import pandas as pd
            >>> import numpy as np
            >>> from gators.imputers import NumericsImputer
            >>> X = pd.DataFrame(
            ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']})
            >>> obj = NumericsImputer(strategy='mean')
            >>> obj.fit_transform(X)
                  A    B  C
            0  0.10  1.0  z
            1  0.20  2.0  a
            2  0.15  1.5  a

        - impute selected numerical columns

            >>> import pandas as pd
            >>> import numpy as np
            >>> from gators.imputers import NumericsImputer
            >>> X = pd.DataFrame(
            ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']})
            >>> obj = NumericsImputer(strategy='mean', columns=['A'])
            >>> obj.fit_transform(X)
                  A    B  C
            0  0.10  1.0  z
            1  0.20  2.0  a
            2  0.15  NaN  a

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.imputers import NumericsImputer
    >>> X = ks.DataFrame({'A': [0.1, 0.2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = NumericsImputer(strategy='mean')
    >>> obj.fit_transform(X)
          A  B
    0  0.10  z
    1  0.20  a
    2  0.15  a

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gators.imputers import NumericsImputer
    >>> X = pd.DataFrame({'A': [0.1, 0.2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = NumericsImputer(strategy='mean')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.1, 'z'],
           [0.2, 'a'],
           [0.15000000000000002, 'a']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.imputers import NumericsImputer
    >>> X = ks.DataFrame({'A': [0.1, 0.2, np.nan], 'B': ['z', 'a', 'a']})
    >>> obj = NumericsImputer(strategy='mean')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.1, 'z'],
           [0.2, 'a'],
           [0.15000000000000002, 'a']], dtype=object)

    See Also
    --------
    gators.imputers.IntImputer
        Impute integer columns.
    gators.imputers.FloatImputer
        Impute float columns.
    gators.imputers.ObjectImputer
        Impute categorical columns.

    """

    def __init__(self, strategy: str, value: float = None, columns: List[str] = None):
        _BaseImputer.__init__(self, strategy, value, columns)
        if strategy not in ["constant", "mean", "median"]:
            raise ValueError(
                """`strategy` should be "constant", ,"mean"
                     or "median" for NumericsImputer Transformer."""
            )
        if strategy == "constant" and not isinstance(value, float):
            raise TypeError(
                """`value` should be a float
                for the NumericsImputer class"""
            )

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "NumericsImputer":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
            'NumericsImputer': Instance of itself.
        """
        self.check_dataframe(X)
        if not self.columns:
            self.columns = util.get_datatype_columns(X, float)
        if not self.columns:
            warnings.warn(
                """`X` does not contain numerical columns,
                `NumericsImputer` is not needed"""
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
        self.statistics_values = np.array(list(self.statistics.values()))
        return self

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
        self.check_dataframe(X)
        if isinstance(X, pd.DataFrame):
            return X.fillna(self.statistics)
        for col, val in self.statistics.items():
            X[col] = X[col].replace({np.nan: val})
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the numpy ndarray X.

        Parameters
        ----------
        X (np.ndarray): Input ndarray.

        Returns
        -------
        np.ndarray:
            Transformed NumPy array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        if X.dtype == object:
            return float_imputer_object(
                X, self.statistics_values.astype(object), self.idx_columns
            )
        return float_imputer(X, self.statistics_values, self.idx_columns)
