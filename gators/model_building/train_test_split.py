from typing import Tuple, Union

import databricks.koalas as ks
import pandas as pd

from ..transformers import TransformerXY
from ..util import util


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

    * transform with `pandas`

        - ordered split

        >>> import pandas as pd
        >>> import numpy as np
        >>> from gators.model_building import TrainTestSplit
        >>> X = pd.DataFrame(np.arange(24).reshape(8, 3), columns=list('ABC'))
        >>> y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')
        >>> test_ratio = 0.5
        >>> obj = TrainTestSplit(test_ratio=test_ratio, strategy='ordered')
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

        - random split

        >>> import pandas as pd
        >>> import numpy as np
        >>> from gators.model_building import TrainTestSplit
        >>> X = pd.DataFrame(np.arange(24).reshape(8, 3), columns=list('ABC'))
        >>> y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')
        >>> test_ratio = 0.5
        >>> obj = TrainTestSplit(test_ratio=test_ratio, strategy='random')
        >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
        >>> X_train
            A   B   C
        0   0   1   2
        3   9  10  11
        4  12  13  14
        5  15  16  17
        >>> X_test
            A   B   C
        6  18  19  20
        2   6   7   8
        1   3   4   5
        7  21  22  23
        >>> y_train
        0    0
        3    0
        4    1
        5    2
        Name: TARGET, dtype: int64
        >>> y_test
        6    0
        2    2
        1    1
        7    1
        Name: TARGET, dtype: int64

        - stratified split

        >>> import pandas as pd
        >>> import numpy as np
        >>> from gators.model_building import TrainTestSplit
        >>> X = pd.DataFrame(np.arange(24).reshape(8, 3), columns=list('ABC'))
        >>> y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')
        >>> test_ratio = 0.5
        >>> obj = TrainTestSplit(test_ratio=test_ratio, strategy='stratified')
        >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
        >>> X_train
           A  B  C
        0  0  1  2
        1  3  4  5
        2  6  7  8
        >>> X_test
            A   B   C
        6  18  19  20
        3   9  10  11
        7  21  22  23
        4  12  13  14
        5  15  16  17
        >>> y_train
        0    0
        1    1
        2    2
        Name: TARGET, dtype: int64
        >>> y_test
        6    0
        3    0
        7    1
        4    1
        5    2
        Name: TARGET, dtype: int64

    * transform with `koalas`

        - ordered split

        >>> import databricks.koalas as ks
        >>> import numpy as np
        >>> from gators.model_building import TrainTestSplit
        >>> X = ks.DataFrame(np.arange(24).reshape(8, 3), columns=list('ABC'))
        >>> y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')
        >>> test_ratio = 0.5
        >>> obj = TrainTestSplit(test_ratio=test_ratio, strategy='ordered')
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

        - random split

        >>> import databricks.koalas as ks
        >>> import numpy as np
        >>> from gators.model_building import TrainTestSplit
        >>> X = ks.DataFrame(np.arange(24).reshape(8, 3), columns=list('ABC'))
        >>> y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')
        >>> test_ratio = 0.5
        >>> obj = TrainTestSplit(test_ratio=test_ratio, strategy='random')
        >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
        >>> X_train
            A   B   C
        0   0   1   2
        7  21  22  23
        3   9  10  11
        2   6   7   8
        >>> X_test
            A   B   C
        6  18  19  20
        5  15  16  17
        1   3   4   5
        4  12  13  14
        >>> y_train
        0    0
        7    1
        3    0
        2    2
        Name: TARGET, dtype: int64
        >>> y_test
        6    0
        5    2
        1    1
        4    1
        Name: TARGET, dtype: int64

        - stratified split

        >>> import databricks.koalas as ks
        >>> import numpy as np
        >>> from gators.model_building import TrainTestSplit
        >>> X = ks.DataFrame(np.arange(24).reshape(8, 3), columns=list('ABC'))
        >>> y = ks.Series([0, 1, 2, 0, 1, 2, 0, 1], name='TARGET')
        >>> test_ratio = 0.5
        >>> obj = TrainTestSplit(test_ratio=test_ratio, strategy='stratified')
        >>> X_train, X_test, y_train, y_test = obj.transform(X, y)
        >>> X_train
            A   B   C
        0   0   1   2
        3   9  10  11
        7  21  22  23
        2   6   7   8
        >>> X_test
            A   B   C
        6  18  19  20
        1   3   4   5
        4  12  13  14
        5  15  16  17
        >>> y_train
        0    0
        3    0
        7    1
        2    2
        Name: TARGET, dtype: int64
        >>> y_test
        6    0
        1    1
        4    1
        5    2
        Name: TARGET, dtype: int64

    """

    def __init__(self, test_ratio: float, strategy: str, random_state: int = 0):
        if not isinstance(strategy, str):
            raise TypeError("`strategy` must be a string.")
        if not isinstance(test_ratio, float):
            raise TypeError("`test_ratio` must be a float.")
        if not isinstance(random_state, int):
            raise TypeError("`random_state` must be an int.")
        if strategy not in ["ordered", "random", "stratified"]:
            raise ValueError("`strategy` not implemented.")
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.strategy = strategy

    def transform(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series],
    ) -> Tuple[
        Union[pd.DataFrame, ks.DataFrame],
        Union[pd.DataFrame, ks.DataFrame],
        Union[pd.Series, ks.Series],
        Union[pd.Series, ks.Series],
    ]:
        """Transform dataframe and series.

        Parameters
        ----------
        X: Union[pd.DataFrame, ks.DataFrame]
            Dataframe.
        y: np.ndarray
            Labels
        test_ratio: float
            Ratio of data points used for the test set.

        Returns
        --------
        Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.DataFrame, ks.DataFrame],
              Union[pd.Series, ks.Series], Union[pd.Series, ks.Series]]
            Train-Test split.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        if self.strategy == "ordered":
            return self.ordered_split(X, y)
        y_name = y.name
        x_name = X.index.name
        if isinstance(X, ks.DataFrame):
            X["index"] = X.index
        Xy = X.join(y)

        if self.strategy == "random":
            Xy_train, Xy_test = self.random_split(Xy, x_name)
        else:
            Xy_train, Xy_test = self.stratified_split(Xy, x_name, y_name)
        return (
            Xy_train.drop(y_name, axis=1),
            Xy_test.drop(y_name, axis=1),
            Xy_train[y_name],
            Xy_test[y_name],
        )

    def ordered_split(
        self, X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]
    ) -> Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.DataFrame, ks.DataFrame]]:
        """Perform random split.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Dataframe
        y : Union[pd.Series, ks.Series]
            Series
        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]:
            Train set.
        Union[pd.DataFrame, ks.DataFrame]:
            Test set.
        """
        n_samples = X.shape[0]
        n_test = int(self.test_ratio * n_samples)
        n_train = n_samples - n_test
        return X.head(n_train), X.tail(n_test), y.head(n_train), y.tail(n_test)

    def random_split(
        self, Xy: Union[pd.DataFrame, ks.DataFrame], x_name: str
    ) -> Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.DataFrame, ks.DataFrame]]:
        """Perform random split.

        Parameters
        ----------
        Xy : Union[pd.DataFrame, ks.DataFrame]
            Dataframe.
        x_name: str
            Dataframe index name.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]:
            Train set.
        Union[pd.DataFrame, ks.DataFrame]:
            Test set.
        """

        if isinstance(Xy, ks.DataFrame):
            self.test_ratio -= 1e-6
            Xy_train, Xy_test = Xy.to_spark().randomSplit(
                [1.0 - self.test_ratio, self.test_ratio], seed=self.random_state
            )
            Xy_train = Xy_train.to_koalas()
            Xy_test = Xy_test.to_koalas()
            Xy_train.set_index("index", drop=True, inplace=True)
            Xy_train.index.name = x_name
            Xy_test.set_index("index", drop=True, inplace=True)
            Xy_test.index.name = x_name
        else:
            Xy_test = Xy.sample(frac=self.test_ratio, random_state=self.random_state)
            Xy_train = Xy.drop(Xy_test.index)
        return Xy_train, Xy_test

    def stratified_split(
        self, Xy: Union[pd.DataFrame, ks.DataFrame], x_name: str, y_name: str
    ) -> Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.DataFrame, ks.DataFrame]]:
        """Perform stratified split.

        Parameters
        ----------
        Xy : Union[pd.DataFrame, ks.DataFrame]
            Dataframe.
        x_name: str
            Dataframe index name.
        y_name: str
            Target column name.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]:
            Train set.
        Union[pd.DataFrame, ks.DataFrame]:
            Test set.
        """
        y_values = Xy[y_name].value_counts(normalize=True)
        Xy_test_list = []
        Xy_train_list = []
        if isinstance(Xy, ks.DataFrame):
            self.test_ratio -= 1e-6
            for label, ratio in y_values.iteritems():
                Xy_label = Xy[Xy[y_name] == label]
                Xy_train_label, Xy_test_label = Xy_label.to_spark().randomSplit(
                    [1.0 - self.test_ratio, self.test_ratio], seed=self.random_state
                )
                Xy_train_label = Xy_train_label.to_koalas()
                Xy_test_label = Xy_test_label.to_koalas()
                Xy_train_label.set_index("index", drop=True, inplace=True)
                Xy_train_label.index.name = x_name
                Xy_test_label.set_index("index", drop=True, inplace=True)
                Xy_test_label.index.name = x_name
                Xy_test_list.append(Xy_test_label)
                Xy_train_list.append(Xy_train_label)
            return util.concat(Xy_train_list, axis=0), util.concat(Xy_test_list, axis=0)

        for label, ratio in y_values.iteritems():
            Xy_label = Xy[Xy[y_name] == label]
            Xy_label_test = Xy_label.sample(
                frac=self.test_ratio, random_state=self.random_state
            )
            Xy_test_list.append(Xy_label_test)
            Xy_train_list.append(Xy_label.drop(Xy_label_test.index))
        return util.concat(Xy_train_list, axis=0), util.concat(Xy_test_list, axis=0)
