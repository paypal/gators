# License: Apache-2.0
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from feature_gen import cluster_statistics

from ._base_feature_generation import _BaseFeatureGeneration


class ClusterStatistics(_BaseFeatureGeneration):
    """Create new columns based on statistics done at the row level.

    The data should be composed of numerical columns only.
    Use `gators.encoders` to replace the categorical columns by
    numerical ones before using `ClusterStatistics`.

    Parameters
    ----------
    clusters_dict :Dict[str, List[str]]
        Dictionary of clusters of features.
    column_names : List[str], default to None.
        List of new column names
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    ---------
    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_generation import ClusterStatistics
    >>> X = pd.DataFrame({'A': [9, 9, 7], 'B': [3, 4, 5], 'C': [6, 7, 8]})
    >>> clusters_dict = {'cluster_1': ['A', 'B'], 'cluster_2': ['A', 'C']}
    >>> obj = ClusterStatistics(clusters_dict=clusters_dict).fit(X)
    >>> obj.fit_transform(X)
         A    B    C  cluster_1__mean  cluster_1__std  cluster_2__mean  cluster_2__std
    0  9.0  3.0  6.0              6.0        4.242641              7.5        2.121320
    1  9.0  4.0  7.0              6.5        3.535534              8.0        1.414214
    2  7.0  5.0  8.0              6.0        1.414214              7.5        0.707107

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import ClusterStatistics
    >>> X = ks.DataFrame({'A': [9, 9, 7], 'B': [3, 4, 5], 'C': [6, 7, 8]})
    >>> clusters_dict = {'cluster_1': ['A', 'B'], 'cluster_2': ['A', 'C']}
    >>> obj = ClusterStatistics(clusters_dict=clusters_dict).fit(X)
    >>> obj.fit_transform(X)
         A    B    C  cluster_1__mean  cluster_1__std  cluster_2__mean  cluster_2__std
    0  9.0  3.0  6.0              6.0        4.242641              7.5        2.121320
    1  9.0  4.0  7.0              6.5        3.535534              8.0        1.414214
    2  7.0  5.0  8.0              6.0        1.414214              7.5        0.707107

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_generation import ClusterStatistics
    >>> X = pd.DataFrame({'A': [9, 9, 7], 'B': [3, 4, 5], 'C': [6, 7, 8]})
    >>> clusters_dict = {'cluster_1': ['A', 'B'], 'cluster_2': ['A', 'C']}
    >>> obj = ClusterStatistics(clusters_dict=clusters_dict)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[9.        , 3.        , 6.        , 6.        , 4.24264069,
            7.5       , 2.12132034],
           [9.        , 4.        , 7.        , 6.5       , 3.53553391,
            8.        , 1.41421356],
           [7.        , 5.        , 8.        , 6.        , 1.41421356,
            7.5       , 0.70710678]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_generation import ClusterStatistics
    >>> X = ks.DataFrame({'A': [9, 9, 7], 'B': [3, 4, 5], 'C': [6, 7, 8]})
    >>> clusters_dict = {'cluster_1': ['A', 'B'], 'cluster_2': ['A', 'C']}
    >>> obj = ClusterStatistics(clusters_dict=clusters_dict)
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
        dtype: type = np.float64,
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
        if column_names is not None and not isinstance(column_names, list):
            raise TypeError("`column_names` should be None or a list.")
        if not column_names:
            column_names = self.get_column_names(clusters_dict)
            column_mapping = {
                **{f"{key}__mean": val for (key, val) in clusters_dict.items()},
                **{f"{key}__std": val for (key, val) in clusters_dict.items()},
            }
        else:
            column_mapping = dict(zip(column_names, clusters_dict.values()))
        columns = list(set([c for s in list(clusters_dict.values()) for c in s]))
        if column_names and 2 * len(clusters_dict) != len(column_names):
            raise ValueError(
                """Length of `column_names` should be
                two times the length of `clusters_dict`."""
            )
        self.check_datatype(dtype, [np.float32, np.float64])
        _BaseFeatureGeneration.__init__(
            self,
            columns=columns,
            column_names=column_names,
            column_mapping=column_mapping,
            dtype=dtype,
        )
        self.clusters_dict = clusters_dict
        self.n_clusters = len(self.clusters_dict)

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "ClusterStatistics":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
        ClusterStatistics
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_dataframe_is_numerics(X)
        self.idx_columns = self.get_idx_columns(X, self.clusters_dict)
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
    ) -> Union[pd.DataFrame, ks.DataFrame]:
        """Transform the dataframe X.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, ks.DataFrame]
            Dataframe with statistics cluster features.
        """
        for i, cols in enumerate(self.clusters_dict.values()):
            X = X.join(X[cols].mean(axis=1).rename(self.column_names[2 * i]))
            X = X.join(X[cols].std(axis=1).rename(self.column_names[2 * i + 1]))
        if isinstance(X, ks.DataFrame):
            return X.astype(self.dtype).sort_index()
        return X.astype(self.dtype)

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
        return cluster_statistics(X.astype(self.dtype), self.idx_columns, self.dtype)

    @staticmethod
    def get_idx_columns(
        X: Union[pd.DataFrame, ks.DataFrame], clusters_dict: Dict[str, List[str]]
    ) -> np.ndarray:
        """Get the column indices of the clusters.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
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
        X : Union[pd.DataFrame, ks.DataFrame].
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
