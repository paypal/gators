# License: Apache-2.0
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import one_hot

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration


class OneHot(_BaseFeatureGeneration):
    """Create new columns based on the one-hot technique.

    Parameters
    ----------
    categories_dict : Dict[str: List[str]].
        keys: columns, values: list of category name.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation import OneHot
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})
    >>> obj = OneHot(categories_dict={'A': ['b', 'c'], 'B': ['z']})
    >>> obj.fit_transform(X)
       A  B  A__onehot__b  A__onehot__c  B__onehot__z
    0  a  z         False         False          True
    1  b  a          True         False         False
    2  c  a         False          True         False

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import OneHot
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})
    >>> obj = OneHot(categories_dict={'A': ['b', 'c'], 'B': ['z']})
    >>> obj.fit_transform(X)
       A  B  A__onehot__b  A__onehot__c  B__onehot__z
    0  a  z         False         False          True
    1  b  a          True         False         False
    2  c  a         False          True         False

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import OneHot
    >>> X = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})
    >>> obj = OneHot(categories_dict={'A': ['b', 'c'], 'B': ['z']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'z', False, False, True],
           ['b', 'a', True, False, False],
           ['c', 'a', False, True, False]], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import OneHot
    >>> X = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['z', 'a', 'a']})
    >>> obj = OneHot(categories_dict={'A': ['b', 'c'], 'B': ['z']})
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
        if column_names is not None and not isinstance(column_names, list):
            raise TypeError("`column_names` should be None or a list.")
        self.categories_dict = categories_dict
        columns = list(set(categories_dict.keys()))
        if not column_names:
            column_names = [
                f"{col}__onehot__{cat}"
                for col, cats in categories_dict.items()
                for cat in cats
            ]
            column_mapping = {
                f"{col}__onehot__{cat}": col
                for col, cats in categories_dict.items()
                for cat in cats
            }
        else:
            column_mapping = {
                name: col for name, col in zip(column_names, categories_dict.keys())
            }
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
            column_mapping=column_mapping,
            dtype=None,
        )
        self.mapping = dict(zip(self.column_names, self.columns))

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ):
        """
        Fit the dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
            y (np.ndarray, optional): labels. Defaults to None.

        Returns
        -------
            OneHot: Instance of itself.
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
        if isinstance(X, pd.DataFrame):
            for name, col, cat in zip(self.column_names, self.columns, self.cats):
                X.loc[:, name] = X[col] == cat
            return X

        for name, col, cat in zip(self.column_names, self.columns, self.cats):
            X = X.assign(dummy=(X[col] == cat)).rename(columns={"dummy": name})
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
        return one_hot(X, self.idx_columns, self.cats)
