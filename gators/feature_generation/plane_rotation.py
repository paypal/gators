# License: Apache-2.0
from math import cos
from math import pi as PI
from math import sin
from typing import List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import plan_rotation
from gators.transformers import Transformer
from gators.util import util

from ._base_feature_generation import _BaseFeatureGeneration


class PlaneRotation(Transformer):
    """Create new columns based on the plane rotation mapping.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `PlaneRotation`.

    Parameters
    ----------
    columns : List[List[str]]
        List of pair-wise columns.
    theta_vec: List[float]
        List of rotation angles.
    dtype : type, default to np.float64
        Numpy dtype of the output data.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gators.feature_generation import PlaneRotation
    >>> X = pd.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [100.0, 125.0]})
    >>> obj = PlaneRotation(
    ... columns=[['X', 'Y'], ['X', 'Z']] , theta_vec=[45.0, 60.0])
    >>> obj.fit_transform(X)
           X      Y      Z  ...  XZ_y_45.0deg  XZ_x_60.0deg  XZ_y_60.0deg
    0  200.0  140.0  100.0  ...    212.132034     13.397460    223.205081
    1  210.0  160.0  125.0  ...    236.880772     -3.253175    244.365335

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import PlaneRotation
    >>> X = ks.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [125.0, 175.0]})
    >>> obj = PlaneRotation(
    ... columns=[['X', 'Y'], ['X', 'Z']] , theta_vec=[45.0])
    >>> obj.fit_transform(X)
           X      Y      Z  XY_x_45.0deg  XY_y_45.0deg  XZ_x_45.0deg  XZ_y_45.0deg
    0  200.0  140.0  125.0     42.426407    240.416306     53.033009    229.809704
    1  210.0  160.0  175.0     35.355339    261.629509     24.748737    272.236111

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import PlaneRotation
    >>> X = pd.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [125.0, 175.0]})
    >>> obj = PlaneRotation(
    ... columns=[['X', 'Y'], ['X', 'Z']], theta_vec=[45.0])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[200.        , 140.        , 125.        ,  42.42640687,
            240.4163056 ,  53.03300859, 229.80970389],
           [210.        , 160.        , 175.        ,  35.35533906,
            261.62950904,  24.74873734, 272.23611076]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import PlaneRotation
    >>> X = ks.DataFrame(
    ... {'X': [200.0, 210.0], 'Y': [140.0, 160.0], 'Z': [125.0, 175.0]})
    >>> obj = PlaneRotation(
    ... columns=[['X', 'Y'], ['X', 'Z']], theta_vec=[45.0])
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[200.        , 140.        , 125.        ,  42.42640687,
            240.4163056 ,  53.03300859, 229.80970389],
           [210.        , 160.        , 175.        ,  35.35533906,
            261.62950904,  24.74873734, 272.23611076]])
    """

    def __init__(
        self, columns: List[List[str]], theta_vec: List[float], dtype: type = np.float64
    ):
        if not isinstance(columns, list):
            raise TypeError("`columns` should be a list.")
        if not isinstance(theta_vec, list):
            raise TypeError("`theta_vec` should be a list.")
        if not columns:
            raise ValueError("`columns` should not be empty.")
        if not any(isinstance(cols, list) for cols in columns):
            raise TypeError("`columns` should be a list of lists.")
        if not all(isinstance(theta, (float, int)) for theta in theta_vec):
            raise TypeError("`theta_vec` should be a list of ints or floats.")
        self.check_datatype(dtype, [np.float32, np.float64])

        column_names = [
            [f"{x}{y}_x_{t}deg", f"{x}{y}_y_{t}deg"]
            for (x, y) in columns
            for t in theta_vec
        ]
        column_names = [c for cols in column_names for c in cols]
        column_mapping = {
            **{f"{x}{y}_x_{t}deg": [x, y] for (x, y) in columns for t in theta_vec},
            **{f"{x}{y}_y_{t}deg": [x, y] for (x, y) in columns for t in theta_vec},
        }
        columns = [c for cols in columns for c in cols]
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
            column_mapping=column_mapping,
            dtype=dtype,
        )
        self.theta_vec = np.array(theta_vec)
        self.cos_theta_vec = np.cos(self.theta_vec * np.pi / 180)
        self.sin_theta_vec = np.sin(self.theta_vec * np.pi / 180)

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "PlaneRotation":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
        PlaneRotation
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.idx_columns_x = util.get_idx_columns(X, self.columns[::2])
        self.idx_columns_y = util.get_idx_columns(X, self.columns[1::2])
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
        for x, y in zip(self.columns[::2], self.columns[1::2]):
            for theta in self.theta_vec:
                cos_theta = cos(theta * PI / 180)
                sin_theta = sin(theta * PI / 180)
                X.loc[:, f"{x}{y}_x_{theta}deg"] = X[x] * cos_theta - X[y] * sin_theta
                X.loc[:, f"{x}{y}_y_{theta}deg"] = X[x] * sin_theta + X[y] * cos_theta
        X[self.column_names] = X[self.column_names].astype(self.dtype)
        return X

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
        return plan_rotation(
            X.astype(self.dtype),
            self.idx_columns_x,
            self.idx_columns_y,
            self.cos_theta_vec,
            self.sin_theta_vec,
        )
