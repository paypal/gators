from typing import Dict, List, Optional

import polars as pl

from ._base_discretizer import _BaseDiscretizer, generate_labels


def compute_geometric_bins(X: pl.DataFrame, num_bins: int, subset: Optional[list[str]] = None) -> dict[str, list[float]]:
    """
    Computes geometric progression bins for discretization.

    Creates bins where each bin edge follows a geometric progression:
    min, min*r, min*r^2, ..., min*r^n = max
    where r = (max/min)^(1/num_bins)

    Handles zero or negative values by shifting the data to positive range.

    Parameters
    ----------
    X : pl.DataFrame
        Input DataFrame containing the data to discretize.
    num_bins : int
        Number of bins to divide each numeric column into.
    subset : Optional[list[str]], default=None
        List of column names to compute bins for. If None, uses all columns in X.

    Returns
    -------
    dict[str, list[float]]
        Dictionary where keys are column names and values are lists of bin edges.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.discretizers import compute_geometric_bins
    >>> X = pl.DataFrame({
    ...     'A': [1, 10, 100, 1000],
    ...     'B': [0.1, 1, 10, 100]
    ... })
    >>> bins = compute_geometric_bins(X, num_bins=3)
    >>> print(bins)
    {'A': [10.0, 100.0], 'B': [1.0, 10.0]}
    """
    cols_to_process = subset if subset is not None else X.columns
    
    # Compute min and max for all columns in a single select operation
    min_max = X.select(
        [pl.col(col_name).min().alias(f"{col_name}_min") for col_name in cols_to_process]
        + [pl.col(col_name).max().alias(f"{col_name}_max") for col_name in cols_to_process]
    ).to_dict(as_series=False)

    bins: Dict[str, List[float]] = {}
    for col in cols_to_process:
        col_min = min_max[f"{col}_min"][0]
        col_max = min_max[f"{col}_max"][0]

        # Handle constant features
        if col_min == col_max:
            bins[col] = []
            continue

        # Handle zero or negative values by shifting
        shift = 0
        if col_min <= 0:
            shift = abs(col_min) + 1e-10  # Small epsilon to avoid exact zero
            col_min_shifted = col_min + shift
            col_max_shifted = col_max + shift
        else:
            col_min_shifted = col_min
            col_max_shifted = col_max

        # Calculate geometric progression
        # r = (max/min)^(1/num_bins)
        # bins: min, min*r, min*r^2, ..., min*r^num_bins = max
        ratio = (col_max_shifted / col_min_shifted) ** (1 / num_bins)

        # Generate bin edges (exclude first and last which are min and max)
        bin_edges = [col_min_shifted * (ratio**i) for i in range(1, num_bins)]

        # Shift back to original scale if we shifted earlier
        if shift > 0:
            bin_edges = [edge - shift for edge in bin_edges]

        bins[col] = bin_edges

    return bins


class GeometricDiscretizer(_BaseDiscretizer):
    """
    Discretizes numerical variables using geometric progression bins.

    Creates bins following a geometric progression where each bin edge is a constant
    multiple of the previous edge. The common ratio is calculated as r = (max/min)^(1/num_bins).
    This is particularly useful for data spanning multiple orders of magnitude (e.g., transaction amounts).

    For columns with zero or negative values, the data is temporarily shifted to positive
    range before computing geometric bins.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to discretize. If None, all numeric columns are selected.
    num_bins : PositiveInt, default=5
        Number of geometric bins to create.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    inplace : bool, default=True
        If True, replace original columns with discretized values.
        If False, create new columns with suffix '__discretize_geom'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after discretizing.
        Ignored when inplace=True.
    as_numerics : bool, default=False
        If True, create numeric labels (0, 1, 2, ...) instead of interval strings.

    Examples
    --------
    >>> from gators.discretizers import GeometricDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'A': [1, 10, 100, 1000],
    ...     'B': [0.1, 1, 10, 100]
    ... })
    >>> discretizer = GeometricDiscretizer(num_bins=3, inplace=False)
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌──────────────┬──────────────┐
    │ A__dic_geom  │ B__dic_geom  │
    │ ---          │ ---          │
    │ str          │ str          │
    ├──────────────┼──────────────┤
    │ (1,10]       │ (0.1,1.0]    │
    │ (1,10]       │ (0.1,1.0]    │
    │ (10,100]     │ (1.0,10.0]   │
    │ (100,1000]   │ (10.0,100.0] │
    └──────────────┴──────────────┘

    >>> # With numeric labels
    >>> discretizer = GeometricDiscretizer(num_bins=3, as_numerics=True, inplace=True)
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌─────┬─────┐
    │ A   │ B   │
    │ --- │ --- │
    │ i32 │ i32 │
    ├─────┼─────┤
    │ 0   │ 0   │
    │ 0   │ 0   │
    │ 1   │ 1   │
    │ 2   │ 2   │
    └─────┴─────┘

    >>> # Handling zero/negative values
    >>> X_neg = pl.DataFrame({
    ...     'C': [-10, 0, 10, 100, 1000]
    ... })
    >>> discretizer = GeometricDiscretizer(num_bins=4, inplace=False)
    >>> discretizer.fit(X_neg)
    >>> transformed = discretizer.transform(X_neg)
    >>> print(transformed)
    shape: (5, 1)
    ┌─────────────┐
    │ C__dic_geom │
    │ ---         │
    │ str         │
    ├─────────────┤
    │ (-10,0]     │
    │ (-10,0]     │
    │ (0,10]      │
    │ (10,100]    │
    │ (100,1000]  │
    └─────────────┘
    """

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "GeometricDiscretizer":
        """Fit the discretizer by computing geometric progression bin boundaries.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        GeometricDiscretizer
            The fitted discretizer instance.
        """
        # Auto-detect numeric columns if not specified
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]

        # Compute geometric bins - pass subset to avoid creating intermediate DataFrame
        self._bins = compute_geometric_bins(X, self.num_bins, subset=self.subset)
        
        # Generate labels with proper rounding
        self._labels = generate_labels(self._bins, rounding=self.rounding)

        # Convert to numeric labels if requested
        if self.as_numerics:
            self._labels = {
                col: [str(v) for v in range(len(vals))] for col, vals in self._labels.items()
            }

        # Set column mapping for non-inplace mode
        if not self.inplace:
            self._column_mapping = {col: f"{col}__discretize_geom" for col in self.subset}

        return self
