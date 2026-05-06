from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
from pydantic import PositiveInt, field_validator

from ._base_discretizer import _BaseDiscretizer, generate_labels


class QuantileDiscretizer(_BaseDiscretizer):
    """
    Flexible quantile-based discretizer with explicit quantile control.

    Creates bins based on specified quantiles, allowing fine-grained control over
    bin boundaries. More flexible than EqualSizeDiscretizer as you can specify
    custom quantiles. Handles skewed distributions well by ensuring each bin
    contains similar numbers of samples.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to discretize. If None, all numeric columns are selected.
    num_bins : PositiveInt, default=5
        Number of quantile-based bins to create (used if quantiles not specified).
        Ignored if quantiles parameter is provided.
    quantiles : Optional[List[float]], default=None
        Explicit list of quantiles (0.0-1.0) to use as bin boundaries.
        If None, equally-spaced quantiles are generated based on num_bins.
        Example: [0.25, 0.5, 0.75] creates quartile bins.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    inplace : bool, default=True
        If True, replace original columns with discretized values.
        If False, create new columns with suffix '__dic_quantile'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after discretizing.
        Ignored when inplace=True.
    as_numerics : bool, default=False
        If True, create numeric labels (0, 1, 2, ...) instead of interval strings.
    handle_duplicates : str, default='drop'
        How to handle duplicate quantile values:

        - 'drop': Remove duplicate bin edges (recommended for low variance data)
        - 'raise': Raise error if duplicates are found

    Examples
    --------
    **Example 1: Quartiles (default behavior with num_bins=4)**

    >>> from gators.discretizers import QuantileDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    ...     'income': [20000, 25000, 30000, 35000, 40000, 50000,
    ...                60000, 70000, 80000, 90000, 100000, 120000]
    ... })
    >>> discretizer = QuantileDiscretizer(
    ...     subset=['age', 'income'],
    ...     num_bins=4,
    ...     drop_columns=True
    ... )
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (12, 2)
    ┌─────────────────┬──────────────────┐
    │ age__dic_quant… ┆ income__dic_qua… │
    │ ---             ┆ ---              │
    │ str             ┆ str              │
    ├─────────────────┼──────────────────┤
    │ (-inf,33.75]    ┆ (-inf,33750.0]   │
    │ (-inf,33.75]    ┆ (-inf,33750.0]   │
    │ ...             ┆ ...              │
    └─────────────────┴──────────────────┘

    **Example 2: Custom quantiles (deciles)**

    >>> discretizer = QuantileDiscretizer(
    ...     subset=['income'],
    ...     quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ...     drop_columns=False
    ... )
    >>> discretizer.fit(X)
    >>> transformed = discretizer.transform(X)

    **Example 3: Asymmetric quantiles for skewed data**

    >>> discretizer_skewed = QuantileDiscretizer(
    ...     subset=['income'],
    ...     quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    ...     drop_columns=True
    ... )
    >>> discretizer_skewed.fit(X)
    >>> transformed = discretizer_skewed.transform(X)

    **Example 4: Tertiles (3 bins)**

    >>> discretizer_tertile = QuantileDiscretizer(
    ...     subset=['age'],
    ...     quantiles=[0.333, 0.667],
    ...     drop_columns=True
    ... )
    >>> discretizer_tertile.fit(X)
    >>> transformed = discretizer_tertile.transform(X)
    """

    quantiles: Optional[List[float]] = None
    handle_duplicates: str = "drop"

    @field_validator("quantiles")
    def check_quantiles(cls, quantiles):
        if quantiles is not None:
            if len(quantiles) == 0:
                raise ValueError("quantiles list cannot be empty")
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("all quantiles must be between 0 and 1 (exclusive)")
            if quantiles != sorted(quantiles):
                raise ValueError("quantiles must be in ascending order")
        return quantiles

    @field_validator("handle_duplicates")
    def check_handle_duplicates(cls, handle_duplicates):
        if handle_duplicates not in ["drop", "raise"]:
            raise ValueError("handle_duplicates must be 'drop' or 'raise'")
        return handle_duplicates

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "QuantileDiscretizer":
        """Fit the discretizer by computing quantile-based bin boundaries.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        QuantileDiscretizer
            The fitted discretizer instance.
        """
        # Identify numeric columns if not specified
        if not self.subset:
            self.subset = [
                col
                for col, dtype in X.schema.items()
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
            ]

        # Determine quantiles to use
        if self.quantiles is not None:
            quantiles_to_use = self.quantiles
        else:
            # Generate equally-spaced quantiles based on num_bins
            quantiles_to_use = np.linspace(0, 1, self.num_bins + 1)[1:-1].tolist()

        # Learn bins for each column
        self._bins = {}

        # Build expressions to compute all quantiles for all columns in a single pass
        quantile_expressions = []
        for col in self.subset:
            for i, q in enumerate(quantiles_to_use):
                quantile_expressions.append(pl.col(col).quantile(q).alias(f"{col}__q{i}"))

        # Compute all quantiles in a single select() operation
        if quantile_expressions:
            quantile_dict = X.select(quantile_expressions).to_dict()

            # Parse results back into bins structure
            for col in self.subset:
                quantile_values = []
                for i in range(len(quantiles_to_use)):
                    alias = f"{col}__q{i}"
                    val = quantile_dict[alias][0]  # First (and only) row
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        quantile_values.append(val)

                # Handle duplicates
                if len(quantile_values) != len(set(quantile_values)):
                    if self.handle_duplicates == "raise":
                        raise ValueError(
                            f"Column '{col}' has duplicate quantile values. "
                            "Consider using fewer bins or set handle_duplicates='drop'."
                        )
                    else:
                        # Drop duplicates while maintaining order
                        seen = set()
                        unique_values = []
                        for val in quantile_values:
                            if val not in seen:
                                seen.add(val)
                                unique_values.append(val)
                        quantile_values = unique_values

                self._bins[col] = sorted(quantile_values)

        # Generate labels
        self._labels = generate_labels(self._bins, self.rounding)

        # Create column mapping
        self._column_mapping = {col: f"{col}__dic_quantile" for col in self.subset}

        return self
