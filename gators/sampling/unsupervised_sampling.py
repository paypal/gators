from typing import Dict, Tuple

from ..transformers import TransformerXY
from ..util import util

from gators import DataFrame, Series


class UnsupervisedSampling(TransformerXY):
    """Randomly sample the data and target.

    Parameters
    ----------
    n_samples : int
        Number of samples to keep.

    Examples
    --------
    >>> from gators.sampling import UnsupervisedSampling
    >>> obj = UnsupervisedSampling(n_samples=3)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [0, 3, 6, 9, 12, 15],
    ... 'B': [1, 4, 7, 10, 13, 16],
    ... 'C': [2, 5, 8, 11, 14, 17]}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([0, 0, 1, 1, 2, 3], name='TARGET'), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': [0, 3, 6, 9, 12, 15],
    ... 'B': [1, 4, 7, 10, 13, 16],
    ... 'C': [2, 5, 8, 11, 14, 17]})
    >>> y = ks.Series([0, 0, 1, 1, 2, 3], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': [0, 3, 6, 9, 12, 15],
    ... 'B': [1, 4, 7, 10, 13, 16],
    ... 'C': [2, 5, 8, 11, 14, 17]})
    >>> y = pd.Series([0, 0, 1, 1, 2, 3], name='TARGET')

    The result is a transformed dataframe and series belonging to the same dataframe library.

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
    """

    def __init__(self, n_samples: int):
        if not isinstance(n_samples, int):
            raise TypeError("`n_samples` should be an int.")
        self.n_samples = n_samples

    def transform(self, X: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
        """Fit and transform the dataframe `X` and the series `y`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series
            Input target.

        Returns
        -------
        X : DataFrame
            Sampled dataframe.
        y : Series
            Sampled series.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        frac = self.n_samples / util.get_function(X).shape(X)[0]
        if frac >= 1.0:
            return X, y
        y_name = y.name
        Xy = (
            util.get_function(X)
            .join(X, y.to_frame())
            .sample(frac=round(frac, 3), random_state=0)
        )
        return Xy.drop(y_name, axis=1), Xy[y_name]
