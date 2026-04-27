from typing import Annotated, Dict, List, Optional, Set

import numpy as np
import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


def find_connected_components(adj_list: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find all connected components in an undirected graph represented by an adjacency list.

    Parameters
    ----------
    adj_list : Dict[int, int]
        Dictionary representing the adjacency list of the graph.
        The keys are nodes, and the values are sets of neighboring nodes.

    Returns
    -------
    List[Set[int]]
        A list of sets, where each set represents a connected component of the graph.
        Each set contains the nodes that are part of the connected component.

    Examples
    --------
    >>> adj_list = {
    ...     0: {1, 2},
    ...     1: {0, 3},
    ...     2: {0},
    ...     3: {1},
    ...     4: {5},
    ...     5: {4}
    ... }
    >>> find_connected_components(adj_list)
    [{0, 1, 2, 3}, {4, 5}]
    """
    visited = set()
    components: List[Set[int]] = []

    def Xs(node: int, component: Set[int]) -> None:
        visited.add(node)
        component.add(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                Xs(neighbor, component)

    for node in adj_list:
        if node not in visited:
            component: Set[int] = set()
            Xs(node, component)
            if len(component) > 1:
                components.append(component)

    return components


class CorrelationFilter(_BaseTransformer):
    """Filters out highly correlated numeric columns.

    Identifies groups of highly correlated columns and removes all but one from each group,
    helping to reduce multicollinearity in the dataset.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric columns to consider for correlation filtering. If None, all numeric columns are used.
    max_corr : float
        Maximum allowed absolute correlation between columns. Must be > 0 and <= 1.
        Columns with correlation >= max_corr are considered highly correlated.

    Examples
    --------
    >>> from correlation_filter import CorrelationFilter
    >>> import polars as pl

    >>> X ={'A': [1, 2, 3, 4],
    ...         'B': [4, 3, 2, 1],
    ...         'C': [1, 2, 1, 2],
    ...         'y': [1, 1, 0, 0]}
    >>> X = pl.DataFrame(X)
    >>> # Example 1
    >>> corr_filter = CorrelationFilter(max_corr=0.9)
    >>> _ = corr_filter.fit(X, y)
    >>> result = corr_filter.transform(X)
    >>> result
    shape: (4, 2)
    ┌─────┬─────┐
    │  C  │  y  │
    │ i64 │ i64 │
    ├─────┼─────┤
    │  1  │  1  │
    │  2  │  1  │
    │  1  │  0  │
    │  2  │  0  │
    └─────┴─────┘

    >>> # Example 2
    >>> corr_filter = CorrelationFilter(subset=['A', 'B'], max_corr=1)
    >>> _ = corr_filter.fit(X)
    >>> result = corr_filter.transform(X)
    >>> result
    shape: (4, 4)
    ┌─────┬─────┬─────┬─────┐
    │  A  │  B  │  C  │  y  │
    │ i64 │ i64 │ i64 │ i64 │
    ├─────┼─────┼─────┼─────┤
    │  1  │  4  │  1  │  1  │
    │  2  │  3  │  2  │  1  │
    │  3  │  2  │  1  │  0  │
    │  4  │  1  │  2  │  0  │
    └─────┴─────┴─────┴─────┘
    """

    subset: Optional[List[str]] = None
    max_corr: Annotated[float, Field(gt=0.0, le=1.0)]
    _to_drop: List[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CorrelationFilter":
        """Fit the transformer by identifying highly correlated columns to drop.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        CorrelationFilter
            The fitted transformer instance.
        """
        if not self.subset:
            self.subset = [col for col in X.columns if X[col].dtype.is_numeric()]

        # Filter out constant columns (zero variance) to avoid division by zero warning
        non_constant_cols = [col for col in self.subset if X[col].std() != 0]

        # If no non-constant columns or only one, nothing to filter
        if len(non_constant_cols) <= 1:
            self._to_drop = []
            return self

        _X_corr_np = np.abs(X[non_constant_cols].corr().to_numpy())
        n = _X_corr_np.shape[0]
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)

        rows, cols = np.where((mask) & (_X_corr_np >= self.max_corr))

        if len(rows) == 0:
            self._to_drop = []
            return self

        adj_list: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i, j in zip(rows, cols):
            adj_list[i].add(j)
            adj_list[j].add(i)

        components = find_connected_components(adj_list)

        to_drop = set()
        for component in components:
            sorted_component = sorted(component)
            to_drop.update(sorted_component[1:])

        self._to_drop = [non_constant_cols[i] for i in sorted(to_drop)]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by dropping highly correlated columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with highly correlated columns removed.
        """
        return X.drop(self._to_drop)
