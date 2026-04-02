from typing import Optional

import numpy as np
import polars as pl

from ._base_discretizer import _BaseDiscretizer, generate_labels


def compute_equal_length_bins(X: pl.DataFrame, num_bins: int) -> dict[str, list[float]]:
    """
    Computes equal-length bins for discretization.

    Parameters
    ----------
    X : pl.DataFrame
        Input DataFrame containing the data to discretize.
    num_bins : int
        Number of bins to divide each numeric column into.

    Returns
    -------
    dict[str, list[float]]
        Dictionary where keys are column names and values are lists of bin edges.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.discretizers import compute_equal_length_bins
    >>> X = pl.DataFrame({
    ...     'A': [0.1, 0.2, 0.3, 0.4],
    ...     'B': [10, 20, 30, 40]
    ... })
    >>> bins = compute_equal_length_bins(X, num_bins=3)
    >>> print(bins)
    {'A': [0.2, 0.3], 'B': [20.0, 30.0]}
    """

    min_max = X.select(
        [pl.col(col_name).min().alias(f"{col_name}_min") for col_name in X.columns]
        + [pl.col(col_name).max().alias(f"{col_name}_max") for col_name in X.columns]
    ).to_dict(as_series=False)
    bins = {}
    for col in X.columns:
        bins[col] = np.linspace(min_max[f"{col}_min"][0], min_max[f"{col}_max"][0], num_bins + 1)[
            1:-1
        ].tolist()
    return bins


class EqualLengthDiscretizer(_BaseDiscretizer):
    """
    Discretizes numerical variables using equal-length bins.

    Creates bins with equal width (range) by dividing the data range into
    num_bins intervals of equal length. Good for uniformly distributed data.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to discretize. If None, all numeric columns are selected.
    num_bins : PositiveInt, default=5
        Number of equal-length bins to create.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    inplace : bool, default=True
        If True, replace original columns with discretized values.
        If False, create new columns with suffix '__discretize_length'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after discretizing.
        Ignored when inplace=True.
    as_numerics : bool, default=False
        If True, create numeric labels (0, 1, 2, ...) instead of interval strings.

    Examples
    --------
    >>> from gators.discretizers import EqualLengthDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'A': [0.1, 0.2, 0.2, 0.4],
    ...     'B': [10, 20, 30, 40]
    ... })
    >>> discretizer = EqualLengthDiscretizer(num_bins=3, drop_columns=True)
    >>> discretizer.subset=['A', 'B']
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬───────────────┐
    │ A__dic_length │ B__dic_length │
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
    │ A   │ B   │ A__dic_length │ B__dic_length │
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
    │ A__dic_length │ B__dic_length │
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
    │ A   │ B   │ A__dic_length │
    │ --- │ --- │ ---           │
    │ f64 │ i64 │ str           │
    ├─────┼─────┼───────────────┤
    │ 0.1 │ 10  │ (0.1,0.2]     │
    │ 0.2 │ 20  │ (0.2,0.3]     │
    │ 0.2 │ 30  │ (0.2,0.3]     │
    │ 0.4 │ 40  │ (0.3,0.4]     │
    └─────┴─────┴───────────────┘
    """

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "EqualLengthDiscretizer":
        """Fit the discretizer by computing equal-length bin boundaries.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        EqualLengthDiscretizer
            The fitted discretizer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]

        self._bins = compute_equal_length_bins(X[self.subset], self.num_bins)
        self._labels = generate_labels(self._bins)
        if self.as_numerics:
            self._labels = {
                col: [str(v) for v in range(len(vals))] for col, vals in self._labels.items()
            }
        if not self.inplace:
            self._column_mapping = {col: f"{col}__discretize_length" for col in self.subset}
        return self
