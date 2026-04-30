from typing import Dict, List, Optional

import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

AGGREGATION_FUNCTIONS = ["min", "max", "mean", "median", "std", "range", "sum"]


class RowStatisticsFeatures(_BaseTransformer):
    """
    Generates row-level aggregation features across groups of columns.

    This transformer computes statistics (min, max, mean, median, std, range)
    horizontally across specified column groups for each row. Unlike
    GroupRatioFeatures which aggregates vertically (across rows within groups),
    this computes statistics across columns within each row.

    Importance for Fraud Detection
    -------------------------------
    Row-level aggregation features are valuable in fraud detection because they
    capture relationships and patterns across related features within individual
    transactions. For example:

    - Computing statistics across multiple transaction amounts can reveal unusual
      patterns (e.g., all amounts being identical might indicate scripted fraud)
    - Aggregating across card verification fields can identify inconsistencies
    - Statistics across temporal features can detect velocity anomalies
    - Range calculations can flag suspiciously uniform or extreme value spreads

    These features help models identify transactions where the distribution of
    values across related fields deviates from normal patterns, which is often
    indicative of fraudulent behavior.

    Parameters
    ----------
    column_groups : Dict[str, List[str]]
        Dictionary mapping group names to lists of column names. Each group defines
        a set of columns over which to compute row-level statistics.
        Example: {'card_fields': ['card1', 'card2', 'card3']}
    func : List[str]
        List of aggregation functions to apply. Available options:

        - 'min': Row-wise minimum value
        - 'max': Row-wise maximum value
        - 'mean': Row-wise mean (average)
        - 'median': Row-wise median
        - 'std': Row-wise standard deviation
        - 'range': Row-wise range (max - min)
        - 'sum': Row-wise sum
    drop_columns : bool, default=False
        Whether to drop the original columns after creating aggregation features.
    new_column_names : Optional[List[str]], default=None
        List of custom names for the aggregation columns. If None, uses default
        naming pattern '{group_name}__{func}'. Must have same length as the total
        number of features created (len(column_groups) × len(func)).

    Examples
    --------
    >>> from gators.feature_generation import RowStatisticsFeatures
    >>> import polars as pl

    **Example 1: Single group with multiple aggregations**

    >>> X = pl.DataFrame({
    ...     'A': [9, 9, 7],
    ...     'B': [3, 4, 5],
    ...     'C': [6, 7, 8]
    ... })
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'cluster_1': ['A', 'B']},
    ...     func=['mean', 'std']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (3, 5)
    ┌─────┬─────┬─────┬───────────────────┬──────────────────┐
    │ A   ┆ B   ┆ C   ┆ cluster_1__mean   ┆ cluster_1__std   │
    │ --- ┆ --- ┆ --- ┆ ---               ┆ ---              │
    │ i64 ┆ i64 ┆ i64 ┆ f64               ┆ f64              │
    ╞═════╪═════╪═════╪═══════════════════╪══════════════════╡
    │ 9   ┆ 3   ┆ 6   ┆ 6.0               ┆ 4.242641         │
    │ 9   ┆ 4   ┆ 7   ┆ 6.5               ┆ 3.535534         │
    │ 7   ┆ 5   ┆ 8   ┆ 6.0               ┆ 1.414214         │
    └─────┴─────┴─────┴───────────────────┴──────────────────┘

    **Example 2: Multiple groups with different columns**

    >>> X = pl.DataFrame({
    ...     'A': [9, 9, 7],
    ...     'B': [3, 4, 5],
    ...     'C': [6, 7, 8],
    ...     'D': [1, 2, 3]
    ... })
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={
    ...         'cluster_1': ['A', 'B'],
    ...         'cluster_2': ['C', 'D']
    ...     },
    ...     func=['min', 'max', 'range']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (3, 10)
    ┌─────┬─────┬─────┬─────┬──────────────┬──────────────┬─────────────────┬──────────────┬──────────────┬─────────────────┐
    │ A   ┆ B   ┆ C   ┆ D   ┆ cluster_1__… ┆ cluster_1__… ┆ cluster_1__ran… ┆ cluster_2__… ┆ cluster_2__… ┆ cluster_2__ran… │
    │ --- ┆ --- ┆ --- ┆ --- ┆ ---          ┆ ---          ┆ ---             ┆ ---          ┆ ---          ┆ ---             │
    │ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64          ┆ i64          ┆ i64             ┆ i64          ┆ i64          ┆ i64             │
    ╞═════╪═════╪═════╪═════╪══════════════╪══════════════╪═════════════════╪══════════════╪══════════════╪═════════════════╡
    │ 9   ┆ 3   ┆ 6   ┆ 1   ┆ 3            ┆ 9            ┆ 6               ┆ 1            ┆ 6            ┆ 5               │
    │ 9   ┆ 4   ┆ 7   ┆ 2   ┆ 4            ┆ 9            ┆ 5               ┆ 2            ┆ 7            ┆ 5               │
    │ 7   ┆ 5   ┆ 8   ┆ 3   ┆ 5            ┆ 7            ┆ 2               ┆ 3            ┆ 8            ┆ 5               │
    └─────┴─────┴─────┴─────┴──────────────┴──────────────┴─────────────────┴──────────────┴──────────────┴─────────────────┘

    **Example 3: Using custom column names**

    >>> X = pl.DataFrame({
    ...     'amount1': [100, 200, 150],
    ...     'amount2': [50, 100, 75],
    ...     'amount3': [25, 50, 30]
    ... })
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'amounts': ['amount1', 'amount2', 'amount3']},
    ...     func=['mean', 'std'],
    ...     new_column_names=['avg_amount', 'std_amount']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (3, 5)
    ┌──────────┬──────────┬──────────┬────────────┬────────────┐
    │ amount1  ┆ amount2  ┆ amount3  ┆ avg_amount ┆ std_amount │
    │ ---      ┆ ---      ┆ ---      ┆ ---        ┆ ---        │
    │ i64      ┆ i64      ┆ i64      ┆ f64        ┆ f64        │
    ╞══════════╪══════════╪══════════╪════════════╪════════════╡
    │ 100      ┆ 50       ┆ 25       ┆ 58.333333  ┆ 30.957...  │
    │ 200      ┆ 100      ┆ 50       ┆ 116.666... ┆ 61.914...  │
    │ 150      ┆ 75       ┆ 30       ┆ 85.0       ┆ 49.606...  │
    └──────────┴──────────┴──────────┴────────────┴────────────┘

    **Example 4: Fraud detection use case - card verification fields**

    >>> X = pl.DataFrame({
    ...     'card_cvv_match': [1, 0, 1, 1],
    ...     'card_addr_match': [1, 1, 0, 1],
    ...     'card_zip_match': [1, 1, 1, 0],
    ...     'is_fraud': [0, 1, 1, 1]
    ... })
    >>> # Aggregate verification fields to detect inconsistencies
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'verification': ['card_cvv_match', 'card_addr_match', 'card_zip_match']},
    ...     func=['mean', 'std'],
    ...     drop_columns=False
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.select(['verification__mean', 'verification__std', 'is_fraud'])
    shape: (4, 3)
    ┌─────────────────────┬────────────────────┬──────────┐
    │ verification__mean  ┆ verification__std  ┆ is_fraud │
    │ ---                 ┆ ---                ┆ ---      │
    │ f64                 ┆ f64                ┆ i64      │
    ╞═════════════════════╪════════════════════╪══════════╡
    │ 1.0                 ┆ 0.0                ┆ 0        │
    │ 0.666667            ┆ 0.471405           ┆ 1        │
    │ 0.666667            ┆ 0.471405           ┆ 1        │
    │ 0.666667            ┆ 0.471405           ┆ 1        │
    └─────────────────────┴────────────────────┴──────────┘
    # Notice: legitimate transaction has perfect verification (mean=1, std=0)
    # Fraudulent transactions show inconsistent verification patterns
    """

    column_groups: Dict[str, List[str]]
    func: List[str]
    drop_columns: bool = False
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("func")
    def check_func(cls, func):
        for f in func:
            if f not in AGGREGATION_FUNCTIONS:
                raise ValueError(
                    f"{f} is not in the predefined list of aggregation functions: {AGGREGATION_FUNCTIONS}"
                )
        return func

    @field_validator("column_groups")
    def check_column_groups(cls, column_groups):
        if not column_groups:
            raise ValueError("column_groups cannot be empty")
        for key, val in column_groups.items():
            if not isinstance(val, list):
                raise TypeError(f"column_groups['{key}'] must be a list")
            if len(val) < 2:
                raise ValueError(
                    f"column_groups['{key}'] must contain at least 2 columns for row-level aggregation"
                )
        return column_groups

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            column_groups = info.data.get("column_groups", {})
            func = info.data.get("func", [])
            expected_length = len(column_groups) * len(func)
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match the total number of features created ({expected_length})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "RowStatisticsFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        RowStatisticsFeatures
            Fitted transformer instance.
        """
        default_names = []
        for group_name in self.column_groups.keys():
            for f in self.func:
                default_names.append(f"{group_name}__{f}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating row-level aggregation features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with row-level aggregation features.
        """
        new_columns = []
        columns_to_drop = set()

        for group_name, cols in self.column_groups.items():
            for f in self.func:
                default_name = f"{group_name}__{f}"
                new_col_name = self._column_mapping[default_name]

                # Create row-wise aggregation expression
                if f == "mean":
                    expr = pl.concat_list(cols).list.mean().alias(new_col_name)
                elif f == "std":
                    expr = pl.concat_list(cols).list.std().alias(new_col_name)
                elif f == "median":
                    expr = pl.concat_list(cols).list.median().alias(new_col_name)
                elif f == "min":
                    expr = pl.concat_list(cols).list.min().alias(new_col_name)
                elif f == "max":
                    expr = pl.concat_list(cols).list.max().alias(new_col_name)
                elif f == "range":
                    expr = (
                        pl.concat_list(cols).list.max() - pl.concat_list(cols).list.min()
                    ).alias(new_col_name)
                elif f == "sum":
                    expr = pl.concat_list(cols).list.sum().alias(new_col_name)

                new_columns.append(expr)

            if self.drop_columns:
                columns_to_drop.update(cols)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(list(columns_to_drop))

        return X
