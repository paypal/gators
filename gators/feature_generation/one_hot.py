# License: Apache-2.0
from typing import Dict, List

import numpy as np

from feature_gen import one_hot

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class OneHot(_BaseFeatureGeneration):
    """Create new columns based on the one-hot technique.

    Parameters
    ----------
    categories_dict : Dict[str: List[str]].
        keys: columns, values: list of category name.

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import OneHot
    >>> categories_dict = {'A': ['b', 'c'], 'B': ['z']}
    >>> obj = OneHot(categories_dict=categories_dict)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
       A  B  A__onehot__b  A__onehot__c  B__onehot__z
    0  a  z         False         False          True
    1  b  a          True         False         False
    2  c  a         False          True         False

    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'z', False, False, True],
           ['b', 'a', True, False, False],
           ['c', 'a', False, True, False]], dtype=object)

    """

    def __init__(
        self, categories_dict: Dict[str, List[str]], column_names: List[str] = None
    ):
        if not isinstance(categories_dict, dict):
            raise TypeError("`categories_dict` should be a dict.")
        if column_names is not None and not isinstance(
            column_names, (list, np.ndarray)
        ):
            raise TypeError("`column_names` should be None or a list.")
        self.categories_dict = categories_dict
        columns = list(set(categories_dict.keys()))
        if not column_names:
            column_names = [
                f"{col}__onehot__{cat}"
                for col, cats in categories_dict.items()
                for cat in cats
            ]
        columns = [col for col, cats in categories_dict.items() for cat in cats]
        n_cats = sum([len(cat) for cat in categories_dict.values()])
        if column_names and n_cats != len(column_names):
            raise ValueError(
                "Length of `clusters_dict` and `column_names` should match."
            )

        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )
        self.mapping = dict(zip(self.column_names, self.columns))

    def fit(self, X: DataFrame, y: Series = None):
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
            y (np.ndarray, optional): labels. Defaults to None.

        Returns
        -------
        self : OneHot
            Instance of itself.
        """
        self.check_dataframe(X)
        self.cats = np.array(
            [cat for cats in self.categories_dict.values() for cat in cats]
        ).astype(object)
        cols_flatten = np.array(
            [col for col, cats in self.categories_dict.items() for cat in cats]
        )
        self.idx_columns = util.get_idx_columns(X, cols_flatten)
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
        util.get_function(X).set_option("compute.ops_on_diff_frames", True)
        for name, col, cat in zip(self.column_names, self.columns, self.cats):
            X[name] = X[col] == cat
        util.get_function(X).set_option("compute.ops_on_diff_frames", False)
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
        return one_hot(X, self.idx_columns, self.cats)
