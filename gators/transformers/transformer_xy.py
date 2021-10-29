# License: Apache-2.0
from abc import ABC, abstractmethod
from typing import Union

import databricks.koalas as ks
import pandas as pd


class TransformerXY(ABC):
    """Abstract **gators** class to transform both X and y.

    Examples
    ---------

    >>> from gators.transformers import TransformerXY
    >>> class FirsRows(TransformerXY):
    ...     def transform(self, X, y):
    ...         return X.head(1), y.head(1)

    * transform with pandas

    >>> import pandas as pd
    >>> X, y = FirsRows().transform(
    ...     X=pd.DataFrame({'A':[1, 2], 'B':[3, 4]}),
    ...     y=pd.Series([0, 1],name='TARGET'))
    >>> X
       A  B
    0  1  3
    >>> y
    0    0
    Name: TARGET, dtype: int64

    * transform with `koalas`

    >>> import databricks.koalas as ks
    >>> X, y = FirsRows().transform(
    ...     X=ks.DataFrame({'A':[1, 2], 'B':[3, 4]}),
    ...     y=ks.Series([0, 1],name='TARGET'))
    >>> X
       A  B
    0  1  3
    >>> y
    0    0
    Name: TARGET, dtype: int64

    """

    @abstractmethod
    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]
    ):
        """Fit and Transform the dataframes `X`ad `y`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : Union[pd.Series, ks.Series]
            None.

        Returns
        -------
        Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.Series, ks.Series]]
            Transformed dataframes.
        """

    @staticmethod
    def check_dataframe(X: Union[pd.DataFrame, ks.DataFrame]):
        """Validate dataframe.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        """
        if not isinstance(X, (pd.DataFrame, ks.DataFrame)):
            raise TypeError(
                """`X` should be a pandas dataframe or a koalas dataframe."""
            )
        for c in X.columns:
            if not isinstance(c, str):
                raise TypeError("Column names of `X` should be of type str.")

    @staticmethod
    def check_y(X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]):
        """Validate target.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Dataframe
        y : Union[pd.Series, ks.Series]
            Labels
        """
        if isinstance(X, pd.DataFrame) and (not isinstance(y, pd.Series)):
            raise TypeError('`y` should be a pandas series.')
        if not isinstance(X, pd.DataFrame) and (not isinstance(y, ks.Series)):
            raise TypeError('`y` should be a koalas series.')
        if not isinstance(y.name, str):
            raise TypeError("Name of `y` should be a str.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Length of `X` and `y` should match.")
