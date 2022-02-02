# License: Apache-2.0
import warnings


import numpy as np
import pandas as pd

from imputer import object_imputer

from ..util import util
from ._base_imputer import _BaseImputer

from gators import DataFrame, Series


class ObjectImputer(_BaseImputer):
    """Impute the categorical columns using the strategy passed by the user.

    Parameters
    ----------
    strategy : str
        Imputation strategy.

        Supported imputation strategies are:

            - 'constant'
            - 'most_frequent'

    value : str, default None.
        Imputation value used for `strategy=constant`.

    Examples
    ---------
    >>> from gators.imputers import ObjectImputer

    The imputation can be done for the selected categorical columns

    >>> obj = ObjectImputer(strategy='most_frequent', columns=['A'])

    or for all the categorical columns

    >>> obj = ObjectImputer(strategy='most_frequent')

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'A': ['a', 'b', 'a', None],
    ... 'B': ['c', 'c', 'd', None],
    ... 'C': [0, 1, 2, np.nan]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> X = ks.DataFrame(
    ... {'A': ['a', 'b', 'a', None],
    ... 'B': ['c', 'c', 'd', None],
    ... 'C': [0, 1, 2, np.nan]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame(
    ... {'A': ['a', 'b', 'a', None],
    ... 'B': ['c', 'c', 'd', None],
    ... 'C': [0, 1, 2, np.nan]})

    The result is a transformed dataframe belonging to the same dataframe library.

    * imputation done for the selected columns:

    >>> obj = ObjectImputer(strategy='most_frequent', columns=['A'])
    >>> obj.fit_transform(X)
       A     B    C
    0  a     c  0.0
    1  b     c  1.0
    2  a     d  2.0
    3  a  None  NaN

    * imputation done for all the columns:

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


    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame(
    ... {'A': ['a', 'b', 'a', None],
    ... 'B': ['c', 'c', 'd', None],
    ... 'C': [0, 1, 2, np.nan]})
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'c', 0.0],
           ['b', 'c', 1.0],
           ['a', 'd', 2.0],
           ['a', 'c', nan]], dtype=object)

    See Also
    --------
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

    def fit(self, X: DataFrame, y: Series = None) -> "ObjectImputer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'ObjectImputer'
            Instance of itself.
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
        self.statistics = self.compute_statistics(X=X, value=self.value)
        self.statistics_np = np.array(list(self.statistics.values())).astype(object)
        return self

    def transform_numpy(self, X: Series, y: Series = None):
        """Transform the NumPy array X.

        Parameters
        ----------
        X :np.ndarray:
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        return object_imputer(X, self.statistics_np, self.idx_columns)
