from typing import Tuple, Union

import databricks.koalas as ks
import pandas as pd

from ..transformers import TransformerXY
from ..util import util


class SupervisedSampling(TransformerXY):
    """Sample each class depending on its occurrence.

    For example, in a stronly imbalanced binary classification problem,
    only the majority is undersampled.

    Parameters
    ----------
    n_samples : int
        Number of samples to keep

    Examples
    --------
    * pandas transform

    >>> import pandas as pd
    >>> from gators.sampling import SupervisedSampling
    >>> X = pd.DataFrame({
    ...     'A': {0: 0, 1: 3, 2: 6, 3: 9, 4: 12, 5: 15},
    ...     'B': {0: 1, 1: 4, 2: 7, 3: 10, 4: 13, 5: 16},
    ...     'C': {0: 2, 1: 5, 2: 8, 3: 11, 4: 14, 5: 17}})
    >>> y = pd.Series([0, 0, 1, 1, 2, 3], name='TARGET')
    >>> obj = SupervisedSampling(n_samples=3)
    >>> X, y = obj.transform(X, y)
    >>> X
        A   B   C
    1   3   4   5
    3   9  10  11
    4  12  13  14
    5  15  16  17
    >>> y
    1    0
    3    1
    4    2
    5    3
    Name: TARGET, dtype: int64

    * koalas transform

    >>> import pandas as ks
    >>> from gators.sampling import SupervisedSampling
    >>> X = ks.DataFrame({
    ...     'A': {0: 0, 1: 3, 2: 6, 3: 9, 4: 12, 5: 15},
    ...     'B': {0: 1, 1: 4, 2: 7, 3: 10, 4: 13, 5: 16},
    ...     'C': {0: 2, 1: 5, 2: 8, 3: 11, 4: 14, 5: 17}})
    >>> y = ks.Series([0, 0, 1, 1, 2, 3], name='TARGET')
    >>> obj = SupervisedSampling(n_samples=3)
    >>> X, y = obj.transform(X, y)
    >>> X
        A   B   C
    1   3   4   5
    3   9  10  11
    4  12  13  14
    5  15  16  17
    >>> y
    1    0
    3    1
    4    2
    5    3
    Name: TARGET, dtype: int64

    """

    def __init__(self, n_samples: int):
        if not isinstance(n_samples, int):
            raise TypeError("`n_samples` should be an int.")
        self.n_samples = n_samples
        self.frac_vec = pd.Series([])

    def transform(
        self, X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]
    ) -> Tuple[Union[pd.DataFrame, ks.DataFrame], Union[pd.Series, ks.Series]]:
        """Fit and transform the dataframe `X` and the series `y`.

        Parameters
        ----------
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
        y_name = y.name
        self.frac_vec = self.compute_sampling_fractions(y, self.n_samples)
        Xy = X.join(y)
        Xy_sampled = self.sample_dataframe(Xy=Xy, frac_vec=self.frac_vec, y_name=y_name)
        return Xy_sampled.drop(y_name, axis=1), Xy_sampled[y_name]

    @staticmethod
    def compute_sampling_fractions(
        y: Union[pd.Series, ks.Series], n_samples: int
    ) -> pd.Series:
        """Compute the sampling fractions.

        Parameters
        ----------
        y : Union[pd.Series, ks.Series]
            Target values.
        n_samples : int
            Number of samples to keep.

        Returns
        -------
        pd.Series
            Fraction to keep for each label.
        """
        class_count = y.value_counts()
        n_classes = len(class_count)
        n_samples_per_class = int(n_samples / n_classes)
        mask = class_count > n_samples_per_class
        n_classes_ = mask.sum()
        n_samples_ = n_samples - class_count[~mask].sum()
        frac_vec = n_samples_ / (class_count * n_classes_)
        if isinstance(frac_vec, ks.Series):
            frac_vec = frac_vec.to_pandas()
        frac_vec[frac_vec > 1] = 1.0
        frac_vec = frac_vec.mask(frac_vec < 1.0 / n_samples, 1.0 / n_samples)
        return frac_vec

    @staticmethod
    def sample_dataframe(
        Xy: Union[pd.DataFrame, ks.DataFrame], frac_vec: pd.Series, y_name: str
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Sample dataframe.

        Parameters
        ----------
        Xy : Union[pd.DataFrame, ks.DataFrame]
            Dataframe.
        frac_vec : pd.Series
            Fraction to keep for each label.
        y_name : str
            Target name.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Transformed dataframe.
        """
        if isinstance(Xy, pd.DataFrame):
            Xy_sampled = pd.DataFrame(columns=Xy.columns)
        else:
            Xy_sampled = ks.DataFrame(columns=Xy.columns)
        for c, frac in frac_vec.iteritems():
            if frac == 1:
                Xy_sampled = util.concat([Xy_sampled, Xy[Xy[y_name] == int(c)]], axis=0)
            else:
                Xy_sampled = util.concat(
                    [
                        Xy_sampled,
                        Xy[Xy[y_name] == int(c)].sample(frac=frac, random_state=0),
                    ],
                    axis=0,
                )
        return Xy_sampled.astype(Xy.dtypes.to_dict())
