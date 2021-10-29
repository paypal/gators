# License: Apache-2.0
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers import Transformer

LIST_NUMPY_NUM_DTYPES = [
    bool,
    int,
    "np.int8",
    "np.int16",
    "np.int32",
    "np.int64",
    "np.uint8",
    "np.uint16",
    "np.uint32",
    "np.uint64",
    float,
    "np.float16",
    "np.float32",
    "np.float64",
]


class ConvertColumnDatatype(Transformer):
    """Set the datatype of the selected columns to a given datatype.

    Parameters
    ----------
    columns : List[str]
        List of columns.
    datatype: type
        Datatype to use.

    Examples
    --------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.converter import ConvertColumnDatatype
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = ConvertColumnDatatype(columns=['A'], datatype=float)
    >>> obj.fit_transform(X)
         A  B
    0  1.0  1
    1  2.0  1
    2  3.0  1

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.converter import ConvertColumnDatatype
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = ConvertColumnDatatype(columns=['A'], datatype=float)
    >>> obj.fit_transform(X)
         A  B
    0  1.0  1
    1  2.0  1
    2  3.0  1

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.converter import ConvertColumnDatatype
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = ConvertColumnDatatype(columns=['A'], datatype=float)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 1.],
           [2., 1.],
           [3., 1.]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.converter import ConvertColumnDatatype
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> obj = ConvertColumnDatatype(columns=['A'], datatype=float)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 1.],
           [2., 1.],
           [3., 1.]])

    """

    def __init__(self, columns: List[str], datatype: type):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if type(datatype) != type and not str(datatype).startswith("datetime"):
            raise TypeError("`datatype` should be a type.")
        self.columns = columns
        self.datatype = datatype

    def fit(
        self, X: Union[pd.DataFrame, ks.DataFrame], y=None
    ) -> "ConvertColumnDatatype":
        """Fit the transformer on the dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        SetDatatype: Instance of itself.
        """
        self.check_dataframe(X)
        return self

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame], y=None
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
        X[self.columns] = X[self.columns].astype(self.datatype)
        return X

    def transform_numpy(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.check_array(X)
        if self.datatype in LIST_NUMPY_NUM_DTYPES:
            return X.astype(self.datatype)
        return X.astype(object)
