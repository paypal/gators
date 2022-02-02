# License: Apache-2.0
from typing import List

import numpy as np

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


from gators import DataFrame, Series


class ConvertColumnDatatype(Transformer):
    """Set the datatype of the selected columns to a given datatype.

    Parameters
    ----------
    theta_vec : List[float]
        List of columns.
    datatype : type
        Datatype to use.

    Examples
    --------

    Imports and initialization:

    >>> from gators.data_cleaning import ConvertColumnDatatype
    >>> obj = ConvertColumnDatatype(columns=['A'], datatype=float)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A  B
    0  1.0  1
    1  2.0  1
    2  3.0  1

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 1.],
           [2., 1.],
           [3., 1.]])
    """

    def __init__(self, columns: List[str], datatype: type):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if type(datatype) != type and not str(datatype).startswith("datetime"):
            raise TypeError("`datatype` should be a type.")
        self.columns = columns
        self.datatype = datatype

    def fit(self, X: DataFrame, y: Series = None) -> "ConvertColumnDatatype":
        """Fit the transformer on the dataframe X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : ConvertColumnDatatype
            Instance of itself.
        """
        self.check_dataframe(X)
        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
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
        X[self.columns] = X[self.columns].astype(self.datatype)
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray, y: Series = None) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        if self.datatype in LIST_NUMPY_NUM_DTYPES:
            return X.astype(self.datatype)
        return X.astype(object)
