from math import isnan
from typing import Optional

import numpy as np
import polars as pl

from ._base_discretizer import _BaseDiscretizer, generate_labels


def compute_equal_size_bins(
    X: pl.DataFrame, num_bins: int, subset: Optional[list[str]] = None
) -> dict[str, list[float]]:
    """
    Discretizes numerical variables using an equal size bins.

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
    >>> from gators.discretizers import compute_equal_size_bins
    >>> X = pl.DataFrame({
    ...     'A': [0.1, 0.2, 0.3, 0.4],
    ...     'B': [10, 20, 30, 40]
    ... })
    >>> bins = compute_equal_size_bins(X, num_bins=3)
    >>> print(bins)
    {'A': [0.2, 0.3], 'B': [20.0, 30.0]}
    """
    cols_to_process = subset if subset is not None else X.columns
    percentiles = np.linspace(0, 1, num_bins + 1)[1:-1]

    # Build all quantile expressions in a single pass - optimize order for better cache locality
    expressions = []
    for col_name in cols_to_process:
        for p in percentiles:
            expressions.append(pl.col(col_name).quantile(p).alias(f"{col_name}_{p}"))

    # Single select operation to compute all quantiles
    bins = X.select(expressions).to_dict(as_series=False)

    # Process results - handle duplicates and nulls efficiently
    selected_bins = {}
    for col in cols_to_process:
        # Extract all quantile values for this column
        col_bins = [bins[f"{col}_{p}"][0] for p in percentiles]
        # Filter out NaN/None values and get unique sorted values in one pass
        col_bins = sorted(
            set(b for b in col_bins if b is not None and not (isinstance(b, float) and isnan(b)))
        )
        selected_bins[col] = col_bins

    return selected_bins


class EqualSizeDiscretizer(_BaseDiscretizer):
    """
    Equal-size discretizer.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of column names to discretize. If None, all numeric columns are used.
    num_bins : PositiveInt, default=5
        Number of bins to divide each numeric column into.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    drop_columns : bool, default=True
        If True, drops original columns after discretizing.

    Examples
    --------
    >>> from gators.discretizers import EqualSizeDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'A': [0.1, 0.2, 0.2, 0.4],
    ...     'B': [10, 20, 30, 40]
    ... })
    >>> discretizer = EqualSizeDiscretizer(num_bins=3, drop_columns=True)
    >>> discretizer.subset=['A', 'B']
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬───────────────┐
    │ A__dic_size   │ B__dic_size   │
    │ ---           │ ---           │
    │ str           │ str           │
    ├───────────────┼───────────────┤
    │ (0.1,0.2]     │ (10,20]       │
    │ (0.1,0.2]     │ (20,30]       │
    │ (0.2,0.3]     │ (20,30]       │
    │ (0.3,0.4]     │ (30,40]       │
    └───────────────┴───────────────┘

    >>> discretizer.drop_columns = False
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 4)
    ┌─────┬─────┬───────────────┬───────────────┐
    │ A   │ B   │ A__dic_size   │ B__dic_size   │
    │ --- │ --- │ ---           │ ---           │
    │ f64 │ i64 │ str           │ str           │
    ├─────┼─────┼───────────────┼───────────────┤
    │ 0.1 │ 10  │ (0.1,0.2]     │ (10,20]       │
    │ 0.2 │ 20  │ (0.1,0.2]     │ (20,30]       │
    │ 0.2 │ 30  │ (0.2,0.3]     │ (20,30]       │
    │ 0.4 │ 40  │ (0.3,0.4]     │ (30,40]       │
    └─────┴─────┴───────────────┴───────────────┘

    >>> discretizer.columns = None
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬───────────────┐
    │ A__dic_size   │ B__dic_size   │
    │ ---           │ ---           │
    │ str           │ str           │
    ├───────────────┼───────────────┤
    │ (0.1,0.2]     │ (10,20]       │
    │ (0.1,0.2]     │ (20,30]       │
    │ (0.2,0.3]     │ (20,30]       │
    │ (0.3,0.4]     │ (30,40]       │
    └───────────────┴───────────────┘

    >>> discretizer.subset=['A']
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 3)
    ┌─────┬─────┬───────────────┐
    │ A   │ B   │ A__dic_size   │
    │ --- │ --- │ ---           │
    │ f64 │ i64 │ str           │
    ├─────┼─────┼───────────────┤
    │ 0.1 │ 10  │ (0.1,0.2]     │
    │ 0.2 │ 20  │ (0.2,0.3]     │
    │ 0.2 │ 30  │ (0.2,0.3]     │
    │ 0.4 │ 40  │ (0.3,0.4]     │
    └─────┴─────┴───────────────┘
    """

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "EqualSizeDiscretizer":
        """Fit the discretizer by computing equal-size (quantile-based) bin boundaries.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        EqualSizeDiscretizer
            The fitted discretizer instance.
        """
        # Auto-detect numeric columns if not specified
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]

        # Compute bins - pass subset to avoid creating intermediate DataFrame
        self._bins = compute_equal_size_bins(X, self.num_bins, subset=self.subset)

        # Generate labels with proper rounding
        self._labels = generate_labels(self._bins, self.rounding)

        # Convert to numeric labels if requested
        if self.as_numerics:
            self._labels = {
                col: [str(v) for v in range(len(vals))] for col, vals in self._labels.items()
            }

        # Set column mapping for non-inplace mode
        if not self.inplace:
            self._column_mapping = {col: f"{col}__discretize_size" for col in self.subset}

        return self
