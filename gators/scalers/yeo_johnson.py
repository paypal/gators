# License: Apache-2.0
from typing import Dict

import numpy as np

from scaler import yeo_johnson

from . import StandardScaler
from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class YeoJohnson(Transformer):
    """Scale the data accoring to the Yeo-Johnson transformation.

    .. note::
       It is recommanded to use the StandardScaler transformer before running YeoJohnson.

    Parameters
    ----------
    lambdas_dict : Dict[str, List[float]]
        The keys are the columns, the values are the list of lambdas:

    Examples
    ---------
    Imports and initialization:

    >>> from gators.scalers import YeoJohnson
    >>> lambdas_dict = {'A': 0.8130050344716966, 'B': 1.0431595843133055, 'C': 0.9168245659045446}
    >>> obj = YeoJohnson(lambdas_dict=lambdas_dict)

    The `fit`, `transform`, and `fit_transform` methods accept:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... "A": [3., 1., -3., -1., -3.],
    ... "B": [22.0, 38.0, -26.0, 35.0, 3 - 5.0],
    ... "C": [7.25, 71.2833, -7.925, -53.1, -8.05]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... "A": [3., 1., -3., -1., -3.],
    ... "B": [22.0, 38.0, -26.0, 35.0, 3 - 5.0],
    ... "C": [7.25, 71.2833, -7.925, -53.1, -8.05]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... "A": [3., 1., -3., -1., -3.],
    ... "B": [22.0, 38.0, -26.0, 35.0, 3 - 5.0],
    ... "C": [7.25, 71.2833, -7.925, -53.1, -8.05]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
              A          B          C
    0  2.566505  24.284823   6.459180
    1  0.930950  42.832241  54.132971
    2 -3.524638 -23.431274  -8.961789
    3 -1.075641  39.324310 -68.684587
    4 -3.524638  -1.945019  -9.111836

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[  2.22845745,  26.90617943,   5.7929225 ],
           [  0.87009573,  48.50683594,  41.9899559 ],
           [ -4.21259165, -21.19840813, -10.21141434],
           [ -1.16205084,  44.38453674, -90.64434052],
           [ -4.21259165,  -1.89256322, -10.39319134]])
    """

    def __init__(self, lambdas_dict: Dict[str, float], inplace: bool = True):
        if not isinstance(lambdas_dict, dict):
            raise TypeError("`lambdas_dict` should be a dictionary.")
        if len(lambdas_dict) == 0:
            raise ValueError("Length of `lambdas_dict` should be not zero.")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        self.lambdas_dict = lambdas_dict
        self.inplace = inplace
        self.lambdas_np = np.array(list(lambdas_dict.values())).astype(float)
        self.columns = list(lambdas_dict.keys())

    def fit(self, X: DataFrame, y: Series = None) -> "YeoJonhson":
        """Fit the transformer on the pandas/koalas dataframe X.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'YeoJonhson'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.idx_columns = util.get_idx_columns(X, self.columns)
        self.column_names = self.get_column_names(
            self.inplace, self.columns, "yeojohnson"
        )
        return self

    def transform(self, X):
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
        for col, name in zip(self.columns, self.column_names):
            lmbda = self.lambdas_dict[col]
            if lmbda == 0:
                X[name] = (
                    X[col]
                    .where(X[col] < 0, np.log1p(X[col]))
                    .where(
                        X[col] >= 0, -((-X[col] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
                    )
                )
            elif lmbda == 2:
                X[name] = (
                    X[col]
                    .where(X[col] < 0, ((X[col] + 1) ** lmbda - 1) / lmbda)
                    .where(X[col] >= 0, -np.log1p(-X[col]))
                )
            else:
                X[name] = (
                    X[col]
                    .where(X[col] < 0, ((X[col] + 1) ** lmbda - 1) / lmbda)
                    .where(
                        X[col] >= 0, -((-X[col] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
                    )
                )

        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X :np.ndarray:
            Input array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
        """
        self.check_array(X)
        X_yeo_johnson = yeo_johnson(X.astype(float), self.lambdas_np)
        if self.inplace:
            X[:, self.idx_columns] = X_yeo_johnson
            return X
        return np.concatenate((X, X_yeo_johnson), axis=1)
        # if self.inplace:
        #     return yeo_johnson(X, self.idx_columns, self.lambdas_np)
        # else:
        #     X_yeo_johnson = yeo_johnson(X.copy(), self.idx_columns, self.lambdas_np)
        #     return np.concatenate((X, X_yeo_johnson[:, self.idx_columns]), axis=1)
