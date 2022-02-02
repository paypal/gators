# License: Apache-2.0
from math import cos
from math import pi as PI
from math import sin
from typing import List

import numpy as np

from feature_gen import plan_rotation
from gators.transformers import Transformer
from gators.util import util

from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class PlaneRotation(Transformer):
    """Create new columns based on the plane rotation mapping.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `PlaneRotation`.

    Parameters
    ----------
    columns : List[List[str]]
        List of pair-wise columns.
    theta_vec : List[float]
        List of rotation angles.
    dtype : type, default np.float64
        Numpy dtype of the output data.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import PlaneRotation
    >>> obj = PlaneRotation(
    ... columns=[['X', 'Y'], ['X', 'Z']] , theta_vec=[45.0, 60.0])

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
           X      Y      Z  ...  XZ_y_45.0deg  XZ_x_60.0deg  XZ_y_60.0deg
    0  200.0  140.0  100.0  ...    212.132034     13.397460    223.205081
    1  210.0  160.0  125.0  ...    236.880772     -3.253175    244.365335

    >>> X = pd.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[200.        , 140.        , 100.        ,  42.42640687,
            240.4163056 , -21.24355653, 243.20508076,  70.71067812,
            212.13203436,  13.39745962, 223.20508076],
           [210.        , 160.        , 125.        ,  35.35533906,
            261.62950904, -33.56406461, 261.86533479,  60.1040764 ,
            236.8807717 ,  -3.25317547, 244.36533479]])
    """

    def __init__(self, columns: List[List[str]], theta_vec: List[float]):

        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not isinstance(theta_vec, (list, np.ndarray)):
            raise TypeError("`theta_vec` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if not any(isinstance(cols, list) for cols in columns):
            raise TypeError("`columns` should be a list of lists.")
        if not all(isinstance(theta, (float, int)) for theta in theta_vec):
            raise TypeError("`theta_vec` should be a list of ints or floats.")

        column_names = [
            [f"{x}{y}_x_{t}deg", f"{x}{y}_y_{t}deg"]
            for (x, y) in columns
            for t in theta_vec
        ]
        column_names = [c for cols in column_names for c in cols]
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )
        self.theta_vec = theta_vec
        self.theta_vec_np = np.array(self.theta_vec)
        self.cos_theta_vec = np.cos(self.theta_vec_np * np.pi / 180)
        self.sin_theta_vec = np.sin(self.theta_vec_np * np.pi / 180)

    def fit(self, X: DataFrame, y: Series = None) -> "PlaneRotation":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        PlaneRotation
            Instance of itself.
        """
        self.check_dataframe(X)
        flatten_columns = [c for cols in self.columns for c in cols]
        self.check_dataframe_is_numerics(X[list(set(flatten_columns))])
        self.idx_columns_x = util.get_idx_columns(
            X[flatten_columns], flatten_columns[::2]
        )
        self.idx_columns_y = util.get_idx_columns(
            X[flatten_columns], flatten_columns[1::2]
        )
        self.idx_subarray = util.get_idx_columns(X.columns, flatten_columns)
        self.flatten_columns = flatten_columns
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
        for x, y in zip(self.flatten_columns[::2], self.flatten_columns[1::2]):
            for theta in self.theta_vec_np:
                cos_theta = cos(theta * PI / 180)
                sin_theta = sin(theta * PI / 180)
                X[f"{x}{y}_x_{theta}deg"] = X[x] * cos_theta - X[y] * sin_theta
                X[f"{x}{y}_y_{theta}deg"] = X[x] * sin_theta + X[y] * cos_theta
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
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
        X_new = plan_rotation(
            X[:, self.idx_subarray].astype(np.float64),
            self.idx_columns_x,
            self.idx_columns_y,
            self.cos_theta_vec,
            self.sin_theta_vec,
        )
        return np.concatenate((X, X_new), axis=1)
