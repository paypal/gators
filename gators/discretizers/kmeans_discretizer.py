from typing import Dict, List, Optional

import numpy as np
import polars as pl
from pydantic import PositiveInt, field_validator
from sklearn.cluster import KMeans

from ._base_discretizer import _BaseDiscretizer, generate_labels


class KMeansDiscretizer(_BaseDiscretizer):
    """
    Clustering-based discretizer using k-means to find natural data clusters.

    Uses k-means clustering to identify natural groupings in the data, creating
    bins based on cluster boundaries. This is more effective than equal-length
    binning for non-uniform distributions as it groups similar values together.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to discretize. If None, all numeric columns are selected.
    num_bins : PositiveInt, default=5
        Number of clusters (bins) to create using k-means.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    inplace : bool, default=True
        If True, replace original columns with discretized values.
        If False, create new columns with suffix '__dic_kmeans'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after discretizing.
        Ignored when inplace=True.
    as_numerics : bool, default=False
        If True, create numeric labels (0, 1, 2, ...) instead of interval strings.
    random_state : Optional[int], default=None
        Random state for reproducibility of k-means clustering.
    max_iter : int, default=300
        Maximum number of iterations for k-means algorithm.
    n_init : int, default=10
        Number of times k-means will be run with different centroid seeds.

    Examples
    --------
    **Example: Non-uniform distribution clustering**

    >>> from gators.discretizers import KMeansDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'price': [10, 12, 15, 18, 100, 105, 110, 500, 520, 550],
    ...     'quantity': [1, 2, 3, 4, 5, 10, 12, 15, 20, 25]
    ... })
    >>> discretizer = KMeansDiscretizer(
    ...     subset=['price', 'quantity'],
    ...     num_bins=3,
    ...     drop_columns=True,
    ...     random_state=42
    ... )
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (10, 2)
    ┌─────────────────┬──────────────────┐
    │ price__dic_kme… ┆ quantity__dic_k… │
    │ ---             ┆ ---              │
    │ str             ┆ str              │
    ├─────────────────┼──────────────────┤
    │ (-inf,56.25]    ┆ (-inf,4.5]       │
    │ (-inf,56.25]    ┆ (-inf,4.5]       │
    │ ...             ┆ ...              │
    └─────────────────┴──────────────────┘

    K-means groups similar values: [10-18], [100-110], [500-550].
    This is more meaningful than equal-length bins like [10-190], [190-370], [370-550].

    >>> discretizer.drop_columns = False
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (10, 4)
    ┌───────┬──────────┬─────────────────┬──────────────────┐
    │ price ┆ quantity ┆ price__dic_kme… ┆ quantity__dic_k… │
    │ ---   ┆ ---      ┆ ---             ┆ ---              │
    │ i64   ┆ i64      ┆ str             ┆ str              │
    ├───────┼──────────┼─────────────────┼──────────────────┤
    │ 10    ┆ 1        ┆ (-inf,56.25]    ┆ (-inf,4.5]       │
    │ 12    ┆ 2        ┆ (-inf,56.25]    ┆ (-inf,4.5]       │
    │ ...   ┆ ...      ┆ ...             ┆ ...              │
    └───────┴──────────┴─────────────────┴──────────────────┘
    """

    random_state: Optional[int] = None
    max_iter: int = 300
    n_init: int = 10
    _centroids: Dict[str, np.ndarray] = {}

    @field_validator("max_iter")
    def check_max_iter(cls, max_iter):
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        return max_iter

    @field_validator("n_init")
    def check_n_init(cls, n_init):
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        return n_init

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "KMeansDiscretizer":
        """Fit the discretizer by learning cluster boundaries using k-means.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        KMeansDiscretizer
            The fitted discretizer instance.
        """
        # Identify numeric columns if not specified
        if not self.subset:
            self.subset = [
                col
                for col, dtype in X.schema.items()
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
            ]

        # Learn bins for each column using k-means
        self._bins = {}
        self._centroids = {}

        for col in self.subset:
            # Get column data and handle nulls
            X_col = X.select(pl.col(col).fill_null(pl.col(col).median())).to_numpy().reshape(-1, 1)

            # Check if we have enough unique values for the requested number of bins
            unique_values = np.unique(X_col)
            n_clusters = min(self.num_bins, len(unique_values))

            if n_clusters < 2:
                # Not enough unique values to cluster
                self._bins[col] = []
                self._centroids[col] = unique_values
                continue

            # Fit k-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init,
            )
            kmeans.fit(X_col)

            # Get cluster centers and sort them
            centroids = sorted(kmeans.cluster_centers_.flatten())
            self._centroids[col] = np.array(centroids)

            # Create bin boundaries at midpoints between centroids
            boundaries = []
            for i in range(len(centroids) - 1):
                midpoint = (centroids[i] + centroids[i + 1]) / 2
                boundaries.append(midpoint)

            self._bins[col] = boundaries

        # Generate labels
        self._labels = generate_labels(self._bins, self.rounding)

        # Create column mapping
        self._column_mapping = {col: f"{col}__dic_kmeans" for col in self.subset}

        return self
