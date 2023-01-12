# License: Apache-2.0
import warnings

import numpy as np

from encoder import onehot_encoder

from ..util import util
from ._base_encoder import _BaseEncoder
from .ordinal_encoder import OrdinalEncoder

from gators import DataFrame, Series


class OneHotEncoder(_BaseEncoder):
    """Encode the categorical columns as one-hot numeric columns.

    Parameters
    ----------
    drop_first : bool, default to True.
        Whether to one-hot encode all the categories or all except one.

    inplace : bool, default to True.
        If True, replace in-place the categorical values by numerical ones.
        If False, keep the categorical columns and create new encoded columns.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.encoders import OneHotEncoder
    >>> obj = OneHotEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A__a  A__b  B__c  B__d
    0   1.0   0.0   1.0   0.0
    1   1.0   0.0   0.0   1.0
    2   0.0   1.0   0.0   1.0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 0., 1., 0.],
           [1., 0., 0., 1.],
           [0., 1., 0., 1.]])
    """

    def __init__(self, drop_first=False, inplace=True):
        _BaseEncoder.__init__(self, inplace=inplace)
        self.drop_first = drop_first
        self.ordinal_encoder = OrdinalEncoder()
        self.idx_numerical_columns = np.array([])
        self.column_names = []
        self.numerical_columns = []

    def fit(self, X: DataFrame, y: Series = None) -> "OneHotEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        OneHotEncoder: Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.columns = util.get_datatype_columns(X, object)
        columns = list(X.columns)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.column_names = list(
            util.get_function(X).get_dummies(
                X[self.columns],
                self.columns,
                prefix_sep="__",
                drop_first=self.drop_first,
            )
        )
        self.column_names = sorted(self.column_names)
        object_columns = ["__".join(col.split("__")[:-1]) for col in self.column_names]
        self.idx_columns = util.get_idx_columns(X, object_columns)
        self.idx_columns_to_keep = [
            i
            for i, col in enumerate(columns + self.column_names)
            if col not in self.columns
        ]

        self.cats = np.array([col.split("__")[-1] for col in self.column_names]).astype(
            object
        )
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        if not self.columns:
            return X

        new_series_list = []
        for name in self.column_names:
            dummy_names = name.split("__")
            col = "__".join(dummy_names[:-1])
            cat = dummy_names[-1]
            dummy = (X[col] == cat).astype(float)
            new_series_list.append(dummy.rename(name))

        if self.inplace:
            return util.get_function(X).concat(
                [
                    X.drop(self.columns, axis=1),
                    util.get_function(X).concat(new_series_list, axis=1),
                ],
                axis=1,
            )
        return util.get_function(X).concat(
            [X, util.get_function(X).concat(new_series_list, axis=1)], axis=1
        )

        # for name in self.column_names:
        #     dummy = name.split("__")
        #     col = "__".join(dummy[:-1])
        #     cat = dummy[-1]
        #     X[name] = (X[col] == cat).astype(float)
        # if self.inplace:
        #     X = X.drop(self.columns, axis=1)
        # return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        X : np.ndarray
            Encoded array.
        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        X_encoded = onehot_encoder(X.copy(), self.idx_columns, self.cats)[
            :, self.idx_columns_to_keep
        ]
        if self.inplace:
            return X_encoded.astype(float)
        return np.concatenate((X, X_encoded[:, -len(self.column_names) :]), axis=1)
