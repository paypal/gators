from typing import Tuple

from ..transformers import TransformerXY
from ..util import util

from gators import DataFrame, Series


class TrainTestSplit(TransformerXY):
    """TrainTestSplit class.

    Parameters
    ----------
    test_ratio : float
        Proportion of the dataset to include in the test split.
    strategy : str
        Train/Test split strategy. The possible values are:

        * ordered
        * random
        * stratified

    random_state : int
        Random state.

    Notes
    -----
    Note that the `random` and `stratified` strategies will be give different
    results for pandas and koalas.

    Examples
    --------
    Imports and initialization:

    >>> from gators.sampling import UnsupervisedSampling
    >>> obj = UnsupervisedSampling(n_samples=3)

    The `transform` method accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({
    ... 'A': [0, 3, 6, 9, 12, 15, 18, 21],
    ... 'B': [1, 4, 7, 10, 13, 16, 19, 22],
    ... 'C': [2, 5, 8, 11, 14, 17, 20, 23]}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET'), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({
    ... 'A': [0, 3, 6, 9, 12, 15, 18, 21],
    ... 'B': [1, 4, 7, 10, 13, 16, 19, 22],
    ... 'C': [2, 5, 8, 11, 14, 17, 20, 23]})
    >>> y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ... 'A': [0, 3, 6, 9, 12, 15, 18, 21],
    ... 'B': [1, 4, 7, 10, 13, 16, 19, 22],
    ... 'C': [2, 5, 8, 11, 14, 17, 20, 23]})
    >>> y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')

    The result is a transformed dataframe and series belonging to the same dataframe library.

    - `ordered` strategy:

    >>> obj = TrainTestSplit(test_ratio=0.5, strategy='ordered')
    >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
    >>> X_train
       A   B   C
    0  0   1   2
    1  3   4   5
    2  6   7   8
    3  9  10  11
    >>> X_test
        A   B   C
    4  12  13  14
    5  15  16  17
    6  18  19  20
    7  21  22  23
    >>> y_train
    0    0
    1    1
    2    2
    3    0
    Name: TARGET, dtype: int64
    >>> y_test
    4    1
    5    2
    6    0
    7    1
    Name: TARGET, dtype: int64



    - `random` strategy:

    >>> obj = TrainTestSplit(test_ratio=0.5, strategy='random')
    >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
    >>> X_train
        A   B   C
    6  18  19  20
    2   6   7   8
    1   3   4   5
    7  21  22  23
    >>> X_test
        A   B   C
    0   0   1   2
    3   9  10  11
    4  12  13  14
    5  15  16  17
    >>> y_train
    6    0
    2    2
    1    1
    7    1
    Name: TARGET, dtype: int64
    >>> y_test
    0    0
    3    0
    4    1
    5    2
    Name: TARGET, dtype: int64

    - `stratified` strategy:

    >>> obj = TrainTestSplit(test_ratio=0.5, strategy='stratified')
    >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
    >>> X_train
        A   B   C
    7  21  22  23
    6  18  19  20
    3   9  10  11
    4  12  13  14
    5  15  16  17
    >>> X_test
       A  B  C
    2  6  7  8
    1  3  4  5
    0  0  1  2
    >>> y_train
    7    1
    6    0
    3    0
    4    1
    5    2
    Name: TARGET, dtype: int64
    >>> y_test
    2    2
    1    1
    0    0
    Name: TARGET, dtype: int64
    """

    def __init__(self, test_ratio: float, strategy: str, random_state: int = 0):
        if not isinstance(strategy, str):
            raise TypeError("`strategy` should be a string.")
        if not isinstance(test_ratio, float):
            raise TypeError("`test_ratio` should be a positive float between 0 and 1.")
        if not isinstance(random_state, int):
            raise TypeError("`random_state` should be a positive int.")
        if strategy not in ["ordered", "random", "stratified"]:
            raise ValueError("`strategy` not implemented.")
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.strategy = strategy

    def transform(
        self,
        X: DataFrame,
        y: Series,
    ) -> Tuple[DataFrame, DataFrame, Series, Series]:
        """Transform dataframe and series.

        Parameters
        ----------
        X : DataFrame
            Dataframe.
        y : Series
            Target values.
        test_ratio : float
            Ratio of data points used for the test set.

        Returns
        --------
        X_train : Dataframe
            Train set X data.

        X_test : Dataframe
            Test set X data.

        y_train : Series
            Train set y data.

        y_test : Series
            Test set y data.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        y_name = y.name
        Xy = util.get_function(X).join(X, y)
        if self.strategy == "ordered":
            Xy_train, Xy_test = self.ordered_split(Xy)
        elif self.strategy == "random":
            Xy_train, Xy_test = util.get_function(X).random_split(
                Xy, frac=self.test_ratio, random_state=self.random_state
            )
        else:  # self.strategy == "stratified"
            Xy_train, Xy_test = self.stratified_split(Xy, y_name)
        return (
            Xy_train.drop(y_name, axis=1),
            Xy_test.drop(y_name, axis=1),
            Xy_train[y_name],
            Xy_test[y_name],
        )

    def ordered_split(self, Xy: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Perform random split.

        Parameters
        ----------
        Xy : DataFrame
            Dataframe.

        Returns
        -------
        Xy_train : DataFrame
            Train set.
        Xy_test : DataFrame:
            Test set.
        """
        n_samples = util.get_function(Xy).shape(Xy)[0]
        n_test = int(self.test_ratio * n_samples)
        n_train = n_samples - n_test
        return util.get_function(Xy).head(Xy, n_train), util.get_function(Xy).tail(
            Xy, n_test
        )

    def stratified_split(
        self, Xy: DataFrame, y_name: str
    ) -> Tuple[DataFrame, DataFrame]:
        """Perform stratified split.

        Parameters
        ----------
        Xy : DataFrame
            Dataframe.

        y_name : str
            Target name.

        Returns
        -------
        Xy_train : DataFrame
            Train set.
        Xy_test : DataFrame:
            Test set.
        """
        labels = util.get_function(Xy).to_pandas(Xy[y_name].unique())
        Xy_train, Xy_test = util.get_function(Xy).random_split(
            Xy[Xy[y_name] == int(labels[0])], frac=self.test_ratio, random_state=0
        )
        for t in labels[1:]:
            dummy_train, dummy_test = util.get_function(Xy).random_split(
                Xy[Xy[y_name] == int(t)], frac=self.test_ratio, random_state=0
            )
            Xy_train = util.get_function(Xy).concat([Xy_train, dummy_train])
            Xy_test = util.get_function(Xy).concat([Xy_test, dummy_test])
        return Xy_train.sample(
            frac=1.0, random_state=self.random_state
        ), Xy_test.sample(frac=1.0, random_state=self.random_state)
