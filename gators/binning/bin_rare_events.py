# License: Apache-2.0
import warnings
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from binning import bin_rare_events

from ..transformers.transformer import Transformer
from ..util import util


class BinRareEvents(Transformer):
    """Replace low occurence categories by the value "OTHERS".

    Use `BinRareEvents` to reduce the cardinality
    of high cardinal columns. This transformer is also useful
    to replace unseen categories by a value which is already
    taken it account by the encoders.

    Parameters
    ----------
    min_ratio : float
        Min occurence ratio per category.

    Examples
    ---------

    >>> import pandas as pd
    >>> from gators.binning import BinRareEvents
    >>> obj = BinRareEvents(min_ratio=0.5)
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})
    >>> obj.fit_transform(X)
            A       B
    0       a  OTHERS
    1       a  OTHERS
    2  OTHERS  OTHERS

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.binning import BinRareEvents
    >>> obj = BinRareEvents(min_ratio=0.5)
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})
    >>> obj.fit_transform(X)
            A       B
    0       a  OTHERS
    1       a  OTHERS
    2  OTHERS  OTHERS

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.binning import BinRareEvents
    >>> obj = BinRareEvents(min_ratio=0.5)
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'OTHERS'],
           ['a', 'OTHERS'],
           ['OTHERS', 'OTHERS']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.binning import BinRareEvents
    >>> obj = BinRareEvents(min_ratio=0.5)
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'OTHERS'],
           ['a', 'OTHERS'],
           ['OTHERS', 'OTHERS']], dtype=object)

    """

    def __init__(self, min_ratio: float):
        if not isinstance(min_ratio, float):
            raise TypeError("""`min_ratio` should be a float.""")
        Transformer.__init__(self)
        self.min_ratio = min_ratio
        self.columns = []
        self.idx_columns: np.ndarray = np.array([])
        self.categories_to_keep_np: np.ndarray = None
        self.n_categories_to_keep_np: np.ndarray = None
        self.categories_to_keep_dict: Dict[str, np.ndarray] = {}

    def fit(self, X: Union[pd.DataFrame, ks.DataFrame], y=None) -> "BinRareEvents":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
        BinRareEvents
            Instance of itself.
        """
        self.check_dataframe(X)
        if object not in X.dtypes.to_numpy():
            warnings.warn(
                """`X` does not contain object columns:
                `BinRareEvents` is not needed"""
            )
            return self
        self.columns = util.get_datatype_columns(X, datatype=object)
        self.categories_to_keep_dict = self.compute_categories_to_keep_dict(
            X=X[self.columns],
            min_ratio=self.min_ratio,
        )
        self.categories_to_keep_np = self.get_categories_to_keep_np(
            categories_to_keep_dict=self.categories_to_keep_dict,
        )
        self.n_categories_to_keep_np = self.categories_to_keep_np.shape[0] - (
            self.categories_to_keep_np == None
        ).sum(0)
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
        )
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

        def f(x):
            name = x.name
            if name not in self.categories_to_keep_dict:
                return x
            return x.mask(~x.isin(self.categories_to_keep_dict[name]), "OTHERS")

        return X.apply(f)

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the NumPy array.

        Parameters
        ----------
        X : np.ndarray
            NumPy array.

        Returns
        -------
        np.ndarray
            Transformed NumPy array.

        """
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        if self.categories_to_keep_np.shape[0] == 0:
            X[:, self.idx_columns] = "OTHERS"
            return X
        return bin_rare_events(
            X,
            self.categories_to_keep_np,
            self.n_categories_to_keep_np,
            self.idx_columns,
        )

    @staticmethod
    def compute_categories_to_keep_dict(
        X: Union[pd.DataFrame, ks.DataFrame], min_ratio: float
    ) -> Dict[str, List[str]]:
        """Compute the category frequency.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        min_ratio : float
            Min occurence per category.

        Returns
        -------
            Dict[str, List[str]]: Categories to keep.
        """

        def f(x):
            freq = x.astype("object").value_counts(normalize=True).sort_values()
            freq = freq[freq >= min_ratio]
            return list(freq.index)

        mapping = X.apply(f).to_dict()
        mapping = {
            key: val if isinstance(val, list) else list(val.values())
            for key, val in mapping.items()
        }
        return mapping

    @staticmethod
    def get_categories_to_keep_np(
        categories_to_keep_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Get the categories to keep.

        Parameters
        ----------
        categories_to_keep_dict : Dict[str, np.ndarray])
            Categories to keep.

        Returns
        -------
        np.ndarray
            Categories to keep.
        """
        max_category = max([len(val) for val in categories_to_keep_dict.values()])
        n_columns = len(categories_to_keep_dict)
        categories_to_keep_np = np.empty((max_category, n_columns), dtype="object")
        for i, val in enumerate(categories_to_keep_dict.values()):
            categories_to_keep_np[: len(val), i] = val
        return categories_to_keep_np
