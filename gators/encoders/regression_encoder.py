# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers import Transformer
from ._base_encoder import _BaseEncoder
from .multiclass_encoder import MultiClassEncoder


class RegressionEncoder(_BaseEncoder):
    """Encode the categorical columns with a binary encoder given by the user.

    The encoding is composed in 2 steps:

    * bin the target values using the discretizer passed as argument.
    * apply the `MultiClassEncoder` on the discretized target values.

    Parameters
    ----------
    encoder : Transformer.
        Encoder.
    discretizer: Transformer.
        Discretizer.
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    --------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.binning import QuantileDiscretizer
    >>> from gators.encoders import WOEEncoder
    >>> from gators.encoders import RegressionEncoder
    >>> X = pd.DataFrame({
    ... 'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ... 'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ... 'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ... 'D': [1, 2, 3, 4, 5, 6]})
    >>> y = pd.Series([0.11,  -0.1, 5.55, 233.9, 4.66, 255.1], name='TARGET')
    >>> obj = RegressionEncoder(
    ... encoder=WOEEncoder(),
    ... discretizer=QuantileDiscretizer(n_bins=3, inplace=True))
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
    >>> from gators.binning import QuantileDiscretizer
    >>> from gators.encoders import WOEEncoder
    >>> from gators.encoders import RegressionEncoder
    >>> X = ks.DataFrame({
    ... 'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ... 'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ... 'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ... 'D': [1, 2, 3, 4, 5, 6]})
    >>> y = ks.Series([0.11,  -0.1, 5.55, 233.9, 4.66, 255.1], name='TARGET')
    >>> obj = RegressionEncoder(
    ... encoder=WOEEncoder(),
    ... discretizer=QuantileDiscretizer(n_bins=3, inplace=True))
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
    >>> from gators.binning import QuantileDiscretizer
    >>> from gators.encoders import WOEEncoder
    >>> from gators.encoders import RegressionEncoder
    >>> X = pd.DataFrame({
    ... 'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ... 'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ... 'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ... 'D': [1, 2, 3, 4, 5, 6]})
    >>> y = pd.Series([0.11,  -0.1, 5.55, 233.9, 4.66, 255.1], name='TARGET')
    >>> obj = RegressionEncoder(
    ... encoder=WOEEncoder(),
    ... discretizer=QuantileDiscretizer(n_bins=3, inplace=True))
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
    >>> from gators.binning import QuantileDiscretizer
    >>> from gators.encoders import WOEEncoder
    >>> from gators.encoders import RegressionEncoder
    >>> X = ks.DataFrame({
    ... 'A': ['Q', 'Q', 'Q', 'W', 'W', 'W'],
    ... 'B': ['Q', 'Q', 'W', 'W', 'W', 'W'],
    ... 'C': ['Q', 'Q', 'Q', 'Q', 'W', 'W'],
    ... 'D': [1, 2, 3, 4, 5, 6]})
    >>> y = ks.Series([0.11,  -0.1, 5.55, 233.9, 4.66, 255.1], name='TARGET')
    >>> obj = RegressionEncoder(
    ... encoder=WOEEncoder(),
    ... discretizer=QuantileDiscretizer(n_bins=3, inplace=True))
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

    def __init__(
        self, encoder: Transformer, discretizer: Transformer, dtype: type = np.float64
    ):
        _BaseEncoder.__init__(self, dtype=dtype)
        if not isinstance(discretizer, Transformer):
            raise TypeError("`discretizer` should inherit from _BaseDiscretizer.")
        if not isinstance(encoder, Transformer):
            raise TypeError("`encoder` should be a transformer.")

        self.discretizer = discretizer
        self.multiclass_encoder = MultiClassEncoder(encoder=encoder, dtype=dtype)

    def fit(
        self, X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]
    ) -> "RegressionEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        RegressionEncoder
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.check_regression_target(y)
        y_binned = self.discretizer.fit_transform(y.to_frame())
        self.multiclass_encoder.fit(X, y_binned[y.name].astype(float).astype(int))
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
        return self.multiclass_encoder.transform(X)

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
        return self.multiclass_encoder.transform_numpy(X)
