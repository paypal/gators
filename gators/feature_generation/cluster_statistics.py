# License: Apache-2.0
from typing import Dict, List

import numpy as np

from feature_gen import cluster_statistics

from ..util import util
from ._base_feature_generation import _BaseFeatureGeneration

from gators import DataFrame, Series


class ClusterStatistics(_BaseFeatureGeneration):
    """Create new columns based on statistics done at the row level.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `ClusterStatistics`.

    Parameters
    ----------
    clusters_dict : Dict[str, List[str]]
        Dictionary of clusters of features.
    column_names : List[str], default None.
        List of new column names

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_generation import ClusterStatistics
    >>> clusters_dict = {'cluster_1': ['A', 'B'], 'cluster_2': ['A', 'C']}
    >>> obj = ClusterStatistics(clusters_dict=clusters_dict)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame(
    ... {'A': [9., 9., 7.], 'B': [3., 4., 5.], 'C': [6., 7., 8.]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame(
    ... {'A': [9., 9., 7.], 'B': [3., 4., 5.], 'C': [6., 7., 8.]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ... {'A': [9., 9., 7.], 'B': [3., 4., 5.], 'C': [6., 7., 8.]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B    C  cluster_1__mean  cluster_1__std  cluster_2__mean  cluster_2__std
    0  9.0  3.0  6.0              6.0        4.242641              7.5        2.121320
    1  9.0  4.0  7.0              6.5        3.535534              8.0        1.414214
    2  7.0  5.0  8.0              6.0        1.414214              7.5        0.707107

    >>> X = pd.DataFrame({'A': [9., 9., 7.], 'B': [3., 4., 5.], 'C': [6., 7., 8.]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[9.        , 3.        , 6.        , 6.        , 4.24264069,
            7.5       , 2.12132034],
           [9.        , 4.        , 7.        , 6.5       , 3.53553391,
            8.        , 1.41421356],
           [7.        , 5.        , 8.        , 6.        , 1.41421356,
            7.5       , 0.70710678]])
    """

    def __init__(
        self,
        clusters_dict: Dict[str, List[str]],
        column_names: List[str] = None,
    ):
        if not isinstance(clusters_dict, dict):
            raise TypeError("`clusters_dict` should be a dict.")
        for key, val in clusters_dict.items():
            if not isinstance(val, list):
                raise TypeError("`clusters_dict` values should be a list.")
            if len(val) == 0:
                raise ValueError("`clusters_dict` values should be a not empty list.")
        cluster_length = [len(v) for v in clusters_dict.values()]
        if min(cluster_length) != max(cluster_length):
            raise ValueError("`clusters_dict` values should be lists with same length.")
        if cluster_length[0] == 1:
            raise ValueError(
                """`clusters_dict` values should be
                lists with a length larger than 1."""
            )
        if column_names is not None and not isinstance(
            column_names, (list, np.ndarray)
        ):
            raise TypeError("`column_names` should be None or a list.")
        if not column_names:
            column_names = self.get_column_names(clusters_dict)
        columns = list(set([c for s in list(clusters_dict.values()) for c in s]))
        if column_names and 2 * len(clusters_dict) != len(column_names):
            raise ValueError(
                """Length of `column_names` should be
                two times the length of `clusters_dict`."""
            )
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
        )
        self.clusters_dict = clusters_dict
        self.n_clusters = len(self.clusters_dict)

    def fit(self, X: DataFrame, y: Series = None) -> "ClusterStatistics":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        ClusterStatistics
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X[self.columns])
        self.idx_subarray = util.get_idx_columns(X, self.columns)
        self.idx_columns = self.get_idx_columns(X[self.columns], self.clusters_dict)
        return self

    def transform(
        self,
        X: DataFrame,
    ) -> DataFrame:
        """Transform the dataframe X.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        X : DataFrame
            Dataframe with statistics cluster features.
        """
        self.check_dataframe(X)
        util.get_function(X).set_option("compute.ops_on_diff_frames", True)
        for i, cols in enumerate(self.clusters_dict.values()):
            X[self.column_names[2 * i]] = X[cols].mean(axis=1)
            X[self.column_names[2 * i + 1]] = X[cols].std(axis=1)
        util.get_function(X).set_option("compute.ops_on_diff_frames", False)
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
        X_new = cluster_statistics(
            X[:, self.idx_subarray].astype(np.float64), self.idx_columns
        )
        return np.concatenate((X, X_new), axis=1)

    @staticmethod
    def get_idx_columns(
        X: DataFrame, clusters_dict: Dict[str, List[str]]
    ) -> np.ndarray:
        """Get the column indices of the clusters.

        Parameters
        ----------
        X : DataFrame
            Input data.
        clusters_dict : Dict[str, List[str]]
            Clusters.

        Returns
        -------
        Dict[str, List[str]]
            Column indices of the clusters.
        """
        columns = list(X.columns)
        n_columns = len(columns)
        idx_columns = np.array(
            [
                [i for i in range(n_columns) if columns[i] in cluster_columns]
                for cluster_columns in list(clusters_dict.values())
            ]
        )
        return idx_columns

    @staticmethod
    def get_column_names(clusters_dict: Dict[str, List[str]]) -> List[str]:
        """Get statistics cluster column names.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.

        Returns
        -------
        List[str]
        List of columns.
        """
        column_names = []
        for name in clusters_dict.keys():
            column_names.append(name + "__mean")
            column_names.append(name + "__std")
        return column_names
