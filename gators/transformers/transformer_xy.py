# License: Apache-2.0
from abc import ABC, abstractmethod


from ..util import util

from gators import DataFrame, Series


class TransformerXY(ABC):
    """Abstract **gators** transformer class to transform both X and y.

    Examples
    ---------

    >>> from gators.transformers import TransformerXY
    >>> class FirsRows(TransformerXY):
    ...     def transform(self, X, y):
    ...         return X.head(1), y.head(1)

    * transform with pandas

    >>> import pandas as pd
    >>> X, y = FirsRows().transform(
    ... X=pd.DataFrame({'A':[1, 2], 'B':[3, 4]}),
    ... y=pd.Series([0, 1],name='TARGET'))
    >>> X
       A  B
    0  1  3
    >>> y
    0    0
    Name: TARGET, dtype: int64

    * transform with `koalas`

    >>> import databricks.koalas as ks
    >>> X, y = FirsRows().transform(
    ... X=ks.DataFrame({'A':[1, 2], 'B':[3, 4]}),
    ... y=ks.Series([0, 1],name='TARGET'))
    >>> X
       A  B
    0  1  3
    >>> y
    0    0
    Name: TARGET, dtype: int64

    """

    @abstractmethod
    def transform(self, X: DataFrame, y: Series):
        """Fit and Transform the dataframes `X`ad `y`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series
            None.

        Returns
        -------
        Tuple[DataFrame, Series]
            Transformed dataframes.
        """

    @staticmethod
    def check_dataframe(X: DataFrame):
        """Validate dataframe.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
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
