# License: Apache-2.0
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from typing import List

import numpy as np

from ..util import util

from gators import DataFrame, Series


class Transformer(ABC, BaseEstimator, TransformerMixin):
    """Abstract **gators** transformer class.

    Examples
    ---------

    * A **Gators** transformer.

    >>> import pandas as pd
    >>> import pyspark.pandas as ps
    >>> import numpy as np
    >>> from gators.transformers import Transformer
    >>> class GetFirstColumn(Transformer):
    ...    def fit(self, X, y: Series = None):
    ...        return self
    ...    def transform(self, X: pd.DataFrame):
    ...        return X[[X.columns[0]]]
    ...    def transform_numpy(self, X: np.ndarray):
    ...        return X[:, 0].reshape(-1, 1)
    >>> obj = GetFirstColumn()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = dd.from_pandas(pd.DataFrame({'A':[1, 2], 'B':[3, 4]}), npartitions=1)

    * `koalas` dataframes:

    >>> import pyspark.pandas as ps
    >>> import numpy as np
    >>> X = ps.DataFrame({'A':[1, 2], 'B':[3, 4]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({'A':[1, 2], 'B':[3, 4]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A
    0  1
    1  2

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame({'A':[1, 2], 'B':[3, 4]})
    >>> obj.transform_numpy(X.to_numpy())
    array([[1],
           [2]])
    """

    @abstractmethod
    def fit(self, X: DataFrame, y: Series = None) -> "Transformer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : Transformer
            Instance of itself.
        """

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """

    @abstractmethod
    def transform_numpy(self, X: Series, y: Series = None):
        """Transform the array `X`.

        Parameters
        ----------
        X : np.ndarray
             Array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """

    def set_columns(self, X: DataFrame, include: List[type], suffix: str):
        """Set the columns of the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.
        include : List[type]
            A list of dtypes.
        suffix : str
            Suffix for the column names.

        """
        self.base_columns = list(X.columns)
        if not self.columns:
            self.columns = list(X.select_dtypes(include=include).columns)
        self.column_names = (
            self.columns if self.inplace else [f"{c}__{suffix}" for c in self.columns]
        )
        self.idx_columns = (
            util.get_idx_columns(X.columns, self.columns)
            if self.columns
            else np.array([])
        )

    @staticmethod
    def check_dataframe(X: DataFrame):
        """Validate dataframe.

        Parameters
        ----------
        X : DataFrame
            Dataframe.
        """
        util.get_function(X)
        for c in X.columns:
            if not isinstance(c, str):
                raise TypeError("Column names of `X` should be of type str.")

    @staticmethod
    def check_target(X: DataFrame, y: Series):
        """Validate target.

        Parameters
        ----------
        X : DataFrame
            Dataframe.
        y : Series
            Target values.
        """
        util.get_function(X).raise_y_dtype_error(y)
        if not isinstance(y.name, str):
            raise TypeError("Name of `y` should be a str.")
        shape = util.get_function(X).shape
        if shape(X)[0] != shape(y)[0]:
            raise ValueError("Length of `X` and `y` should match.")

    @staticmethod
    def check_array(X: np.ndarray):
        """Validate array.

        Parameters
        ----------
        X : np.ndarray
             Array.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("`X` should be a NumPy array.")

    @staticmethod
    def get_column_names(inplace: bool, columns: List[str], suffix: str):
        """Return the names of the modified columns.

        Parameters
        ----------
        inplace : bool
            If True return `columns`.
            If False return `columns__suffix`.
        columns : List[str]
            List of columns.
        suffix : str
            Suffix used if `inplace` is False.

        Returns
        -------
        List[str]
            List of column names.
        """
        return columns if inplace else [f"{c}__{suffix}" for c in columns]
