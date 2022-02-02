# License: Apache-2.0
import warnings
from typing import Dict, List

import numpy as np

from binning import bin_rare_events

from ..transformers.transformer import Transformer
from ..util import util

from gators import DataFrame, Series


class BinRareCategories(Transformer):
    """Replace low occurence categories by the value "OTHERS".

    Use `BinRareCategories` to reduce the cardinality
    of high cardinal columns. This transformer is also useful
    to replace unseen categories by a value which is already
    taken it account by the encoders.

    Parameters
    ----------
    min_ratio : float
        Min occurence ratio per category.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.binning import BinRareCategories
    >>> obj = BinRareCategories(min_ratio=0.5)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
            A       B
    0       a  OTHERS
    1       a  OTHERS
    2  OTHERS  OTHERS

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([['a', 'OTHERS'],
           ['a', 'OTHERS'],
           ['OTHERS', 'OTHERS']], dtype=object)
    """

    def __init__(self, min_ratio: float):
        if not isinstance(min_ratio, (int, float)) or min_ratio < 0 or min_ratio > 1:
            raise TypeError(
                """`min_ratio` should be a positive float betwwen 0.0 and 1.0."""
            )
        Transformer.__init__(self)
        self.min_ratio = min_ratio
        self.columns = []
        self.idx_columns: np.ndarray = np.array([])
        self.categories_to_keep_np: np.ndarray = None
        self.n_categories_to_keep_np: np.ndarray = None
        self.categories_to_keep_dict: Dict[str, np.ndarray] = {}

    def fit(self, X: DataFrame, y: Series = None) -> "BinRareCategories":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        BinRareCategories
            Instance of itself.
        """
        self.check_dataframe(X)
        if object not in X.dtypes.to_numpy():
            warnings.warn(
                """`X` does not contain object columns:
                `BinRareCategories` is not needed"""
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
        for col in self.columns:
            X[col] = X[col].mask(
                ~X[col].isin(self.categories_to_keep_dict[col]), "OTHERS"
            )
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Transform the array `X`.

        Parameters
        ----------
        X : np.ndarray
             Array.

        Returns
        -------
        X : np.ndarray
            Transformed array.
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
        X: DataFrame, min_ratio: float
    ) -> Dict[str, List[str]]:
        """Compute the category frequency.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        min_ratio : float
            Min occurence per category.

        Returns
        -------
        mapping : Dict[str, List[str]]
            Categories to keep.
        """
        freq = util.get_function(X).to_pandas(
            util.get_function(X).melt(X).groupby(["variable", "value"]).size() / len(X)
        )
        freq = freq[freq >= min_ratio]
        mapping = {}
        for c in X.columns:
            if c in list(freq.index.get_level_values(0)):
                mapping[c] = list(freq.loc[c].index)
            else:
                mapping[c] = []
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
        categories_to_keep_np : np.ndarray
            Categories to keep.
        """
        max_category = max([len(val) for val in categories_to_keep_dict.values()])
        n_columns = len(categories_to_keep_dict)
        categories_to_keep_np = np.empty((max_category, n_columns), dtype="object")
        for i, val in enumerate(categories_to_keep_dict.values()):
            categories_to_keep_np[: len(val), i] = val
        return categories_to_keep_np
