from typing import Dict, List, Optional

import numpy as np
import polars as pl
from pydantic import PositiveInt

from gators.discretizers._base_discretizer import _BaseDiscretizer, generate_labels


class CustomDiscretizer(_BaseDiscretizer):
    """
    Custom discretizer class.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to discretize. If None, uses all columns in the bins dictionary.
    bins : Dict[str, List[float]]
        Dictionary specifying bin edges for each column. Keys are column names, values are lists of bin boundaries.
        Use -np.inf and np.inf for open-ended bins.
    num_bins : PositiveInt, default=5
        Number of bins (used for validation, not for computation since bins are custom).
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    inplace : bool, default=True
        If True, replace original columns with discretized values.
        If False, create new columns with suffix '__discretize_custom'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after discretizing.
        Ignored when inplace=True.
    as_numerics : bool, default=False
        If True, create numeric labels (0, 1, 2, ...) instead of interval strings.

    Examples
    --------
    >>> from gators.discretizers import CustomDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'A': [0.1, 0.2, 0.2, 0.4],
    ...     'B': [10, 20, 30, 40]
    ... })
    >>> bins = {
    ...     'A': [-np.inf, 0.2, 0.3, np.inf],
    ...     'B': [-np.inf, 20, 30, np.inf]
    ... }
    >>> discretizer = CustomDiscretizer(bins=bins, num_bins=3, drop_columns=True)
    >>> discretizer.subset=['A', 'B']
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬─────────────────────┐
    │ A__discretize │ B__discretize       │
    │ _custom       │ _custom             │
    │ ---           │ ---                 │
    │ str           │ str                 │
    ├───────────────┼─────────────────────┤
    │ (-inf,0.2]    │ (-inf,20.0]         │
    │ (0.2,0.3]     │ (20.0,30.0]         │
    │ (0.2,0.3]     │ (20.0,30.0]         │
    │ (0.3,inf)     │ (30.0,inf)          │
    └───────────────┴─────────────────────┘

    >>> discretizer.drop_columns = False
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 4)
    ┌─────┬─────┬───────────────┬───────────────┐
    │ A   │ B   │ A__discretize │ B__discretize │
    │ --- │ --- │ _custom       │ _custom       │
    │ f64 │ i64 │ str           │ str           │
    ├─────┼─────┼───────────────┼───────────────┤
    │ 0.1 │ 10  │ (-inf,0.2]    │ (-inf,20.0]   │
    │ 0.2 │ 20  │ (0.2,0.3]     │ (20.0,30.0]   │
    │ 0.2 │ 30  │ (0.2,0.3]     │ (20.0,30.0]   │
    │ 0.4 │ 40  │ (0.3,inf)     │ (30.0,inf)    │
    └─────┴─────┴───────────────┴───────────────┘

    >>> discretizer.columns = None
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 2)
    ┌───────────────┬─────────────────────┐
    │ A__discretize │ B__discretize       │
    │ _custom       │ _custom             │
    │ ---           │ ---                 │
    │ str           │ str                 │
    ├───────────────┼─────────────────────┤
    │ (-inf,0.2]    │ (-inf,20.0]         │
    │ (0.2,0.3]     │ (20.0,30.0]         │
    │ (0.2,0.3]     │ (20.0,30.0]         │
    │ (0.3,inf)     │ (30.0,inf)          │
    └───────────────┴─────────────────────┘

    >>> discretizer.subset=['A']
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (4, 3)
    ┌─────┬─────┬───────────────┐
    │ A   │ B   │ A__discretize │
    │ --- │ --- │ _custom       │
    │ f64 │ i64 │ str           │
    ├─────┼─────┼───────────────┤
    │ 0.1 │ 10  │ (-inf,0.2]    │
    │ 0.2 │ 20  │ (0.2,0.3]     │
    │ 0.2 │ 30  │ (0.2,0.3]     │
    │ 0.4 │ 40  │ (0.3,inf)     │
    └─────┴─────┴───────────────┘
    """
    subset: Optional[List[str]] = None
    # num_bins: PositiveInt = 5
    # rounding: PositiveInt = 3
    drop_columns: bool = True
    bins: Dict[str, List[float]]
    _labels: Dict[str, List[str]]
    _column_mapping: Dict[str, str]

    def fit(
        self, X: pl.DataFrame, y: Optional[pl.Series] = None
    ) -> "CustomDiscretizer":
        """Fit the discretizer using predefined custom bins.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        CustomDiscretizer
            The fitted discretizer instance.
        """
        if self.subset is None:
            self.subset = list(self.bins.keys())

        self._bins = self.bins
        self._labels = generate_labels(bins=self.bins)
        if self.as_numerics:
            self._labels = {
                col: np.arange(len(vals)).astype(str)
                for col, vals in self._labels.items()
            }
        if not self.inplace:
            self._column_mapping = {
                col: f"{col}__discretize_custom" for col in self.subset
            }
        return self
