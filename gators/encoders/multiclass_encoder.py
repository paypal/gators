# License: Apache-2.0
import copy
import warnings
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..data_cleaning.drop_columns import DropColumns
from ..transformers.transformer import Transformer
from ..util import util
from ._base_encoder import _BaseEncoder


class MultiClassEncoder(_BaseEncoder):
    """Encode the categorical columns with a binary encoder passed by the user.
    *N* categorical columns are mapped into *N * (n - 1)* numerical columns
    where *n* is the number of classes.

    Parameters
    ----------
    encoder : Transformer
        Binary Encoder.
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    --------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.encoders import MultiClassEncoder
    >>> from gators.encoders import WOEEncoder
    >>> X = pd.DataFrame({
    ...    'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ...    'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ...    'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ...    'D': [1, 2, 3, 4, 5, 6]})
    >>> y = pd.Series([0,  0, 1, 2, 1, 2], name='TARGET')
    >>> obj = MultiClassEncoder(WOEEncoder())
    >>> obj.fit_transform(X, y)
         D  A__TARGET_1_WOEEncoder  B__TARGET_1_WOEEncoder  C__TARGET_1_WOEEncoder  A__TARGET_2_WOEEncoder  B__TARGET_2_WOEEncoder  C__TARGET_2_WOEEncoder
    0  1.0                     0.0                0.000000               -0.405465                0.000000                0.000000               -0.405465
    1  2.0                     0.0                0.000000               -0.405465                0.000000                0.000000               -0.405465
    2  3.0                     0.0                0.693147               -0.405465                0.000000                0.693147               -0.405465
    3  4.0                     0.0                0.693147               -0.405465                1.386294                0.693147               -0.405465
    4  5.0                     0.0                0.693147                0.693147                1.386294                0.693147                0.693147
    5  6.0                     0.0                0.693147                0.693147                1.386294                0.693147                0.693147

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import MultiClassEncoder
    >>> from gators.encoders import WOEEncoder
    >>> X = ks.DataFrame({
    ...    'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ...    'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ...    'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ...    'D': [1, 2, 3, 4, 5, 6]})
    >>> y = ks.Series([0,  0, 1, 2, 1, 2], name='TARGET')
    >>> obj = MultiClassEncoder(WOEEncoder())
    >>> obj.fit_transform(X, y)
         D  A__TARGET_1_WOEEncoder  B__TARGET_1_WOEEncoder  C__TARGET_1_WOEEncoder  A__TARGET_2_WOEEncoder  B__TARGET_2_WOEEncoder  C__TARGET_2_WOEEncoder
    0  1.0                     0.0                0.000000               -0.405465                0.000000                0.000000               -0.405465
    1  2.0                     0.0                0.000000               -0.405465                0.000000                0.000000               -0.405465
    2  3.0                     0.0                0.693147               -0.405465                0.000000                0.693147               -0.405465
    3  4.0                     0.0                0.693147               -0.405465                1.386294                0.693147               -0.405465
    4  5.0                     0.0                0.693147                0.693147                1.386294                0.693147                0.693147
    5  6.0                     0.0                0.693147                0.693147                1.386294                0.693147                0.693147

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.encoders import MultiClassEncoder
    >>> from gators.encoders import WOEEncoder
    >>> X = pd.DataFrame({
    ...    'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ...    'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ...    'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ...    'D': [1, 2, 3, 4, 5, 6]})
    >>> y = pd.Series([0,  0, 1, 2, 1, 2], name='TARGET')
    >>> obj = MultiClassEncoder(WOEEncoder())
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 1.        ,  0.        ,  0.        , -0.40546511,  0.        ,
             0.        , -0.40546511],
           [ 2.        ,  0.        ,  0.        , -0.40546511,  0.        ,
             0.        , -0.40546511],
           [ 3.        ,  0.        ,  0.69314718, -0.40546511,  0.        ,
             0.69314718, -0.40546511],
           [ 4.        ,  0.        ,  0.69314718, -0.40546511,  1.38629436,
             0.69314718, -0.40546511],
           [ 5.        ,  0.        ,  0.69314718,  0.69314718,  1.38629436,
             0.69314718,  0.69314718],
           [ 6.        ,  0.        ,  0.69314718,  0.69314718,  1.38629436,
             0.69314718,  0.69314718]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import MultiClassEncoder
    >>> from gators.encoders import WOEEncoder
    >>> X = ks.DataFrame({
    ...    'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ...    'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ...    'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ...    'D': [1, 2, 3, 4, 5, 6]})
    >>> y = ks.Series([0,  0, 1, 2, 1, 2], name='TARGET')
    >>> obj = MultiClassEncoder(WOEEncoder())
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 1.        ,  0.        ,  0.        , -0.40546511,  0.        ,
             0.        , -0.40546511],
           [ 2.        ,  0.        ,  0.        , -0.40546511,  0.        ,
             0.        , -0.40546511],
           [ 3.        ,  0.        ,  0.69314718, -0.40546511,  0.        ,
             0.69314718, -0.40546511],
           [ 4.        ,  0.        ,  0.69314718, -0.40546511,  1.38629436,
             0.69314718, -0.40546511],
           [ 5.        ,  0.        ,  0.69314718,  0.69314718,  1.38629436,
             0.69314718,  0.69314718],
           [ 6.        ,  0.        ,  0.69314718,  0.69314718,  1.38629436,
             0.69314718,  0.69314718]])
    """

    def __init__(self, encoder: Transformer, dtype: type = np.float64):
        if not isinstance(encoder, Transformer):
            raise TypeError("`encoder` should be a transformer.")
        _BaseEncoder.__init__(self, dtype=dtype)
        self.encoder = encoder
        self.drop_columns = None
        self.label_names = []
        self.encoder_dict = {}
        self.columns = []
        self.idx_columns = np.ndarray([])
        self.column_names = []
        self.column_mapping = {}
        self.name = type(encoder).__name__

    def fit(
        self, X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]
    ) -> "MultiClassEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        MultiClassEncoder
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.check_multiclass_target(y)
        self.columns = util.get_datatype_columns(X, object)
        self.check_nans(X, self.columns)
        self.drop_columns = DropColumns(self.columns).fit(X)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.idx_columns = util.get_idx_columns(
            columns=X.columns,
            selected_columns=self.columns,
        )
        y_name = y.name
        if isinstance(X, pd.DataFrame):
            y_one_hot = pd.get_dummies(y, prefix=y_name)
        else:
            y_one_hot = ks.get_dummies(y, prefix=y_name)
        y_one_hot = y_one_hot.drop(y_one_hot.columns[0], axis=1)
        self.label_names = y_one_hot.columns
        for label_name in self.label_names:
            self.encoder_dict[label_name] = copy.copy(self.encoder)
            self.encoder_dict[label_name].fit(X[self.columns], y_one_hot[label_name])
        return self

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame]
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
            self.idx_columns = np.array([])
            return X
        for i, label_name in enumerate(self.label_names):
            dummy = self.encoder_dict[label_name].transform(X[self.columns].copy())[
                self.encoder_dict[label_name].columns
            ]
            column_names = [f"{col}__{label_name}_{self.name}" for col in dummy.columns]
            dummy.columns = column_names
            self.column_names.extend(column_names)
            for name, col in zip(column_names, self.columns):
                self.column_mapping[name] = col
            X = X.join(dummy, how="inner").sort_index()
        return self.drop_columns.transform(X).astype(self.dtype)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array `X`.

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
        if not self.columns:
            return X
        X_encoded_list = []
        for i, label_name in enumerate(self.label_names):
            dummy = self.encoder_dict[label_name].transform_numpy(
                X[:, self.idx_columns].copy()
            )
            X_encoded_list.append(dummy)
        X_new = np.concatenate(
            [self.drop_columns.transform_numpy(X)] + X_encoded_list, axis=1
        )
        return X_new.astype(self.dtype)
