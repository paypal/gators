from typing import Dict, Tuple, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..transformers import TransformerXY


class UnsupervisedSampling(TransformerXY):
    """Randomly sample the data and target.

    Parameters
    ----------
    n_samples : int
        Number of samples to keep.

    Examples
    --------

    * pandas transform

    >>> import pandas as pd
    >>> from gators.sampling import UnsupervisedSampling
    >>> X = pd.DataFrame({
    ... 'A': {0: 0, 1: 3, 2: 6, 3: 9, 4: 12, 5: 15},
    ... 'B': {0: 1, 1: 4, 2: 7, 3: 10, 4: 13, 5: 16},
    ... 'C': {0: 2, 1: 5, 2: 8, 3: 11, 4: 14, 5: 17}})
    >>> y = pd.Series([0, 0, 1, 1, 2, 3], name='TARGET')
    >>> obj = UnsupervisedSampling(n_samples=3)
    >>> X, y = obj.transform(X, y)
    >>> X
        A   B   C
    5  15  16  17
    2   6   7   8
    1   3   4   5
    >>> y
    5    3
    2    1
    1    0
    Name: TARGET, dtype: int64

    * koalas transform

    >>> import databricks.koalas as ks
    >>> from gators.sampling import UnsupervisedSampling
    >>> X = ks.DataFrame({
    ... 'A': {0: 0, 1: 3, 2: 6, 3: 9, 4: 12, 5: 15},
    ... 'B': {0: 1, 1: 4, 2: 7, 3: 10, 4: 13, 5: 16},
    ... 'C': {0: 2, 1: 5, 2: 8, 3: 11, 4: 14, 5: 17}})
    >>> y = ks.Series([0, 0, 1, 1, 2, 3], name='TARGET')
    >>> obj = UnsupervisedSampling(n_samples=3)
    >>> X, y = obj.transform(X, y)
    >>> X
       A   B   C
    0  0   1   2
    3  9  10  11
    2  6   7   8
    >>> y
    0    0
    3    1
    2    1
    Name: TARGET, dtype: int64

    """

    def __init__(self, n_samples: int):
        if not isinstance(n_samples, int):
            raise TypeError("`n_samples` should be an int.")
        self.n_samples = n_samples

    def transform(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series],
    ) -> Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.Series, ks.Series]]:
        """Fit and transform the dataframe `X` and the series `y`.

        Parameters:
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series]
            Input target.

        Returns
        -------
        Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.Series, ks.Series]]:
            Transformed dataframe and the series.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        frac = self.n_samples / X.shape[0]
        if frac >= 1.0:
            return X, y
        y_name = y.name
        Xy = X.join(y).sample(frac=round(frac, 3), random_state=0)
        return Xy.drop(y_name, axis=1), Xy[y_name]
