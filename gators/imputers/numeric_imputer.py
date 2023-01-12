# License: Apache-2.0
import warnings
from typing import List

import numpy as np

from imputer import float_imputer

from ..util import util
from ._base_imputer import _BaseImputer

from gators import DataFrame, Series


class NumericImputer(_BaseImputer):
    """Impute the numerical columns using the strategy passed by the user.

    Parameters
    ----------
    strategy : str
        Imputation strategy.

        Supported imputation strategies are:

            - 'constant'
            - 'mean'
            - 'median'

    value : str, default None.
        Imputation value used for `strategy=constant`.
    inplace : bool, default True.
        If True, impute in-place.
        If False, create new imputed columns.

    Examples
    ---------

    >>> from gators.imputers import NumericImputer

    >>> bins = {'A':[-np.inf, 0, np.inf], 'B':[-np.inf, 1, np.inf]}

    The imputation can be done for the selected numerical columns

    >>> obj = NumericImputer(strategy='mean', columns=['A'])

    or for all the numerical columns

    >>> obj = NumericImputer(strategy='mean')

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> X = ks.DataFrame(
    ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame(
    ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']})

    The result is a transformed dataframe belonging to the same dataframe library.

    * imputation done for the selected columns:

    >>> obj = NumericImputer(strategy='mean', columns=['A'])
    >>> obj.fit_transform(X)
          A    B  C
    0  0.10  1.0  z
    1  0.20  2.0  a
    2  0.15  NaN  a

    * imputation done for all the columns:

    >>> X = pd.DataFrame(
    ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']})
    >>> obj = NumericImputer(strategy='mean')
    >>> obj.fit_transform(X)
          A    B  C
    0  0.10  1.0  z
    1  0.20  2.0  a
    2  0.15  1.5  a


    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame(
    ... {'A': [0.1, 0.2, np.nan], 'B': [1, 2, np.nan], 'C': ['z', 'a', 'a']})
    >>> obj.transform_numpy(X.to_numpy())
    array([[0.1, 1.0, 'z'],
           [0.2, 2.0, 'a'],
           [0.15000000000000002, 1.5, 'a']], dtype=object)

    See Also
    --------
    gators.imputers.ObjectImputer
        Impute categorical columns.
    """

    def __init__(
        self,
        strategy: str,
        value: float = None,
        columns: List[str] = None,
        inplace: bool = True,
    ):
        _BaseImputer.__init__(self, strategy, value, columns=columns, inplace=inplace)
        if strategy == "constant" and not isinstance(self.value, (int, float)):
            raise TypeError(
                """`value` should be an int or a float
                for the NumericImputer class"""
            )
        self.value = float(self.value) if self.value is not None else None

    def fit(self, X: DataFrame, y: Series = None) -> "NumericImputer":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'NumericImputer'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        if not self.columns:
            self.columns = util.get_datatype_columns(X, float)
            self.columns += util.get_datatype_columns(X, int)
        self.column_names = self.get_column_names(self.inplace, self.columns, "impute")
        if not self.columns:
            warnings.warn(
                """`X` does not contain numerical columns,
                `NumericImputer` is not needed"""
            )
            self.idx_columns = np.array([])
            return self
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        self.statistics = self.compute_statistics(X=X, value=self.value)

        self.statistics_np = np.array(list(self.statistics.values()))
        return self

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array X.

        Parameters
        ----------
        X :np.ndarray:
            Input array.

        Returns
        -------
        X : np.ndarray:
            Transformed array.
        """
        self.check_array(X)
        if isinstance(X[0, 0], np.integer):
            return X
        if self.idx_columns.size == 0:
            return X
        if self.inplace:
            X[:, self.idx_columns] = float_imputer(
                X[:, self.idx_columns].astype(float), self.statistics_np
            )
            return X
        else:
            X_impute = float_imputer(
                X[:, self.idx_columns].copy().astype(float), self.statistics_np
            )
            return np.concatenate((X, X_impute), axis=1)
