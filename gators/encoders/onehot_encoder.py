# License: Apache-2.0
import warnings
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from encoder import onehot_encoder

from ..util import util
from . import OrdinalEncoder
from ._base_encoder import _BaseEncoder


class OneHotEncoder(_BaseEncoder):
    """Encode the categorical columns as one-hot numeric columns.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.encoders import OneHotEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OneHotEncoder()
    >>> obj.fit_transform(X)
       A__b  A__a  B__d  B__c
    0   0.0   1.0   0.0   1.0
    1   0.0   1.0   1.0   0.0
    2   1.0   0.0   1.0   0.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import OneHotEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OneHotEncoder()
    >>> obj.fit_transform(X)
       A__b  A__a  B__d  B__c
    0   0.0   1.0   0.0   1.0
    1   0.0   1.0   1.0   0.0
    2   1.0   0.0   1.0   0.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.encoders import OneHotEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OneHotEncoder()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0., 1., 0., 1.],
           [0., 1., 1., 0.],
           [1., 0., 1., 0.]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import OneHotEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OneHotEncoder()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[0., 1., 0., 1.],
           [0., 1., 1., 0.],
           [1., 0., 1., 0.]])
    """

    def __init__(self, dtype: type = np.float64):
        _BaseEncoder.__init__(self, dtype=dtype)
        self.ordinal_encoder = OrdinalEncoder(dtype=dtype, add_other_columns=False)
        self.idx_numerical_columns = np.array([])
        self.onehot_columns = []
        self.numerical_columns = []
        self.column_mapping = {}

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "OneHotEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
        OneHotEncoder: Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_datatype_columns(X, object)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.check_nans(X, self.columns)
        self.numerical_columns = util.exclude_columns(X.columns, self.columns)
        _ = self.ordinal_encoder.fit(X)
        self.onehot_columns = []
        for key, val in self.ordinal_encoder.mapping.items():
            self.onehot_columns.extend(
                [f"{key}__{self.dtype(c)}" for c in sorted(val.values(), key=int)]
            )
        for key, val in self.ordinal_encoder.mapping.items():
            for k, v in val.items():
                self.column_mapping[f"{key}__{self.dtype(v)}"] = f"{key}__{k}"
        self.all_columns = self.numerical_columns + self.onehot_columns
        self.idx_numerical_columns = util.get_idx_columns(
            X.columns, self.numerical_columns
        )
        self.idx_columns = np.arange(
            len(self.numerical_columns),
            len(self.numerical_columns) + len(self.onehot_columns),
            dtype=int,
        )
        self.idx_columns = np.arange(
            len(self.numerical_columns), len(self.onehot_columns), dtype=int
        )
        self.n_categories_vec = np.empty(len(self.ordinal_encoder.columns), int)
        for i, c in enumerate(self.columns):
            self.n_categories_vec[i] = len(self.ordinal_encoder.mapping[c])

        self.columns_flatten = np.array(
            [
                col
                for col, mapping in self.ordinal_encoder.mapping.items()
                for v in range(len(mapping))
            ]
        )
        self.idx_columns = util.get_idx_columns(X, self.columns_flatten)
        self.idx_columns_to_keep = [
            i
            for i in range(X.shape[1] + self.idx_columns.shape[0])
            if i not in util.get_idx_columns(X, self.columns)
        ]
        self.cats = np.array(
            [
                v
                for col, mapping in self.ordinal_encoder.mapping.items()
                for v in dict(sorted(mapping.items(), key=lambda item: item[1])).keys()
            ]
        ).astype(object)
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
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
        if not self.columns:
            return X
        dummy = X[self.columns].copy()
        X_new = self.ordinal_encoder.transform(X)
        X[self.columns] = dummy
        if isinstance(X, pd.DataFrame):
            X_new = pd.get_dummies(X_new, prefix_sep="__", columns=self.columns)
        else:
            X_new = ks.get_dummies(X_new, prefix_sep="__", columns=self.columns)
        X_new = X_new.reindex(columns=self.all_columns, fill_value=0.0)
        return X_new.rename(columns=self.column_mapping).astype(self.dtype)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the input array.

        Parameters
        ----------
        X  : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray: Encoded array.
        """
        self.check_array(X)
        if len(self.idx_columns) == 0:
            return X
        return onehot_encoder(X, self.idx_columns, self.cats)[
            :, self.idx_columns_to_keep
        ].astype(self.dtype)
