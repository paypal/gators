# License: Apache-2.0
import warnings
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd
from imputer import object_imputer

from ..util import util
from ._base_imputer import _BaseImputer


class ObjectImputer(_BaseImputer):
    """Impute the categorical columns using the strategy passed by the user.

    Parameters
    ----------
    strategy : str
        Imputation strategy.

        Supported imputation strategies are:

            - 'constant'
            - 'most_frequent'

    value : str, default to None.
        Imputation value used for `strategy=constant`.

    Examples
    ---------

    * fit & transform with `pandas`

        - impute all the object columns

            >>> import pandas as pd
            >>> import numpy as np
            >>> from gators.imputers import ObjectImputer
            >>> X = pd.DataFrame(
            ... {'A': ['a', 'b', 'a', None],
            ... 'B': ['c', 'c', 'd', None],
            ... 'C': [0, 1, 2, np.nan]})
            >>> obj = ObjectImputer(strategy='most_frequent')
            >>> obj.fit_transform(X)
               A  B    C
            0  a  c  0.0
            1  b  c  1.0
            2  a  d  2.0
            3  a  c  NaN

        - impute selected object columns

            >>> import pandas as pd
            >>> import numpy as np
            >>> from gators.imputers import ObjectImputer
            >>> X = pd.DataFrame(
            ... {'A': ['a', 'b', 'a', None],
            ... 'B': ['c', 'c', 'd', None],
            ... 'C': [0, 1, 2, np.nan]})
            >>> obj = ObjectImputer(strategy='most_frequent', columns=['A'])
            >>> obj.fit_transform(X)
               A     B    C
            0  a     c  0.0
            1  b     c  1.0
            2  a     d  2.0
            3  a  None  NaN


    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.imputers import ObjectImputer
    >>> X = ks.DataFrame({'A': ['a', 'b', 'a', np.nan], 'B': [0, 1, 2, np.nan]})
    >>> obj = ObjectImputer(strategy='most_frequent')
    >>> obj.fit_transform(X)
       A    B
    0  a  0.0
    1  b  1.0
    2  a  2.0
    3  a  NaN

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gators.imputers import ObjectImputer
    >>> X = ks.DataFrame({'A': ['a', 'b', 'a', np.nan], 'B': [0, 1, 2, np.nan]})
    >>> obj = ObjectImputer(strategy='most_frequent')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 0.0],
           ['b', 1.0],
           ['a', 2.0],
           ['a', nan]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.imputers import ObjectImputer
    >>> X = ks.DataFrame({'A': ['a', 'b', 'a', np.nan], 'B': [0, 1, 2, np.nan]})
    >>> obj = ObjectImputer(strategy='most_frequent')
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 0.0],
           ['b', 1.0],
           ['a', 2.0],
           ['a', nan]], dtype=object)

    See Also
    --------
    gators.imputers.IntImputer
        Impute integer columns.
    gators.imputers.FloatImputer
        Impute float columns.
    gators.imputers.NumericsImputer
        Impute numerical columns.

    """

    def __init__(self, strategy: str, value: str = None, columns=None):
        _BaseImputer.__init__(self, strategy, value, columns=columns)
        if strategy not in ["constant", "most_frequent"]:
            raise ValueError(
                """`strategy` should be "constant" or "most_frequent"
                    for the ObjectImputer Transformer."""
            )
        if strategy == "constant" and not isinstance(value, str):
            raise TypeError(
                """`value` should be a string
                for the ObjectImputer class"""
            )

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "ObjectImputer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
            ObjectImputer: Instance of itself.
        """
        self.check_dataframe(X)
        if not self.columns:
            self.columns = util.get_datatype_columns(X, object)
        if not self.columns:
            warnings.warn(
                """`X` does not contain object columns:
                `ObjectImputer` is not needed"""
            )
            self.idx_columns = np.array([])
            return self
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        self.idx_columns = np.array(util.get_idx_columns(X, self.columns))
        self.statistics = self.compute_statistics(
            X=X,
            columns=self.columns,
            strategy=self.strategy,
            value=self.value,
        )
        self.statistics_values = np.array(list(self.statistics.values())).astype(object)
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
        return object_imputer(X, self.statistics_values, self.idx_columns)
