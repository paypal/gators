from typing import Dict, Tuple

from ..transformers import TransformerXY
from ..util import util

from gators import DataFrame, Series


class SupervisedSampling(TransformerXY):
    """Sample each class depending on the user input.

    Parameters
    ----------
    n_samples : int
        Number of samples to keep

    Examples
    --------
    Imports and initialization:

    >>> from gators.sampling import SupervisedSampling
    >>> obj = SupervisedSampling(frac_dict={0: 0.5, 1:0.5, 2:1, 3:1})

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
    0   3   4   5
    1   9  10  11
    2  12  13  14
    3  15  16  17
    >>> y
    0    0
    1    1
    2    2
    3    3
    Name: TARGET, dtype: int64
    """

    def __init__(self, frac_dict: Dict[str, float], random_state: int = 0):
        if (not isinstance(random_state, int)) or (random_state < 0):
            raise TypeError("`random_state` should be a positive int.")
        if not isinstance(frac_dict, dict):
            raise TypeError("`frac_dict` should be a dict.")
        for k, v in frac_dict.items():
            if not isinstance(k, int):
                raise TypeError("the keys of `frac_dict` should be an int.")
            if (not isinstance(v, (int, float))) or (v > 1 or v < 0):
                raise TypeError(
                    "the values of `frac_dict` should be float beting 0 an 1."
                )
        self.frac_dict = frac_dict
        self.random_state = random_state

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
        y_name = y.name
        index_name = X.index.name

        def f(x, frac_dict, random_state):
            return x.sample(
                frac=float(frac_dict[x[y_name].iloc[0]]), random_state=random_state
            )

        Xy = util.get_function(X).join(X, y.to_frame())
        Xy_sampled = (
            Xy.groupby(y.name)
            .apply(f, self.frac_dict, self.random_state)
            .reset_index(drop=True)
        )
        Xy_sampled.index.name = index_name
        return Xy_sampled.drop(y_name, axis=1), Xy_sampled[y_name]
