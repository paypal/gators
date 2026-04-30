from typing import Dict, List, Optional

import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

AGGREGATION_FUNCTIONS = ["mean", "std", "median", "min", "max", "sum", "count", "range"]


class GroupStatisticsFeatures(_BaseTransformer):
    """
    Generates statistical aggregation features based on group-level computations.

    Unlike GroupRatioFeatures which divides values by group stats, this transformer
    directly adds the group statistics as new columns.

    Parameters
    ----------
    subset : List[str]
        List of numerical column names to aggregate.
    by : List[str]
        List of column names to use for groupby operations. Each column will be used
        for a separate groupby operation (e.g., ['cat1', 'cat2'] creates features
        grouped by cat1 and separate features grouped by cat2).
    func : List[str]
        List of aggregation functions to apply. Available options:
        - 'mean': Group mean
        - 'std': Group standard deviation
        - 'median': Group median
        - 'min': Group minimum
        - 'max': Group maximum
        - 'sum': Group sum
        - 'count': Group count
        - 'range': Group range (max - min)
    drop_columns : bool, default=False
        Whether to drop the original numerical columns after creating statistics.
    new_column_names : Optional[List[str]], default=None
        List of custom names for the statistic columns. If None, uses default naming pattern
        '{agg}_{num_col}__per_{groupby_col}'. Must have same length as the total number
        of features created (subset × by × func).

    Examples
    --------
    >>> from gators.feature_generation import GroupStatisticsFeatures
    >>> import polars as pl

    >>> X ={
    ...     'amount': [100, 200, 150, 300, 250],
    ...     'cat1': ['A', 'A', 'B', 'B', 'A'],
    ...     'cat2': ['X', 'Y', 'X', 'X', 'X']
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Basic group statistics**

    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['mean', 'count']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (5, 5)
    ┌────────┬───────┬───────┬───────────────────────┬────────────────────────┐
    │ amount ┆ cat1  ┆ cat2  ┆ mean_amount__per_cat1 ┆ count_amount__per_cat1 │
    │ ---    ┆ ---   ┆ ---   ┆ ---                   ┆ ---                    │
    │ i64    ┆ str   ┆ str   ┆ f64                   ┆ u32                    │
    ╞════════╪═══════╪═══════╪═══════════════════════╪════════════════════════╡
    │ 100    ┆ A     ┆ X     ┆ 183.333333            ┆ 3                      │
    │ 200    ┆ A     ┆ Y     ┆ 183.333333            ┆ 3                      │
    │ 150    ┆ B     ┆ X     ┆ 225.0                 ┆ 2                      │
    │ 300    ┆ B     ┆ X     ┆ 225.0                 ┆ 2                      │
    │ 250    ┆ A     ┆ X     ┆ 183.333333            ┆ 3                      │
    └────────┴───────┴───────┴───────────────────────┴────────────────────────┘

    **Example 2: Multiple groupby columns**

    >>> X ={
    ...     'amount': [100, 200, 150, 300],
    ...     'cat1': ['A', 'A', 'B', 'B'],
    ...     'cat2': ['X', 'Y', 'X', 'Y']
    ... }
    >>> X = pl.DataFrame(X)
    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1', 'cat2'],
    ...     func=['mean']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['amount', 'cat1', 'cat2', 'mean_amount__per_cat1', 'mean_amount__per_cat2']
    # Creates separate features grouped by cat1 and grouped by cat2

    **Example 3: Multiple func**

    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['mean', 'std', 'min', 'max']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['amount', 'cat1', 'cat2', 'mean_amount__per_cat1', 'std_amount__per_cat1',
     'min_amount__per_cat1', 'max_amount__per_cat1']
    """

    subset: List[str]
    by: List[str]
    func: List[str]
    drop_columns: bool = False
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("func")
    def check_func(cls, func):
        for fun in func:
            if fun not in AGGREGATION_FUNCTIONS:
                raise ValueError(
                    f"{fun} is not in the predefined list of aggregation functions: {AGGREGATION_FUNCTIONS}"
                )
        return func

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            subset = info.data.get("subset", [])
            by = info.data.get("by", [])
            func = info.data.get("func", [])
            expected_length = len(subset) * len(by) * len(func)
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match the total number of features created ({expected_length}). "
                    f"Expected: {len(subset)} numerical columns × {len(by)} groupby columns × {len(func)} func = {expected_length}"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "GroupStatisticsFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        GroupStatisticsFeatures
            Fitted transformer instance.
        """
        default_names = []
        for num_col in self.subset:
            for groupby_col in self.by:
                for fun in self.func:
                    default_names.append(f"{fun}_{num_col}__per_{groupby_col}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating group statistic features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with group statistic features.
        """
        new_columns = []

        for num_col in self.subset:
            for groupby_col in self.by:
                for fun in self.func:
                    default_name = f"{fun}_{num_col}__per_{groupby_col}"
                    new_col_name = self._column_mapping[default_name]

                    # Compute group aggregation for single column
                    agg_functions = {
                        "mean": pl.col(num_col).mean().over(groupby_col),
                        "std": pl.col(num_col).std().over(groupby_col),
                        "median": pl.col(num_col).median().over(groupby_col),
                        "min": pl.col(num_col).min().over(groupby_col),
                        "max": pl.col(num_col).max().over(groupby_col),
                        "sum": pl.col(num_col).sum().over(groupby_col),
                        "count": pl.col(num_col).count().over(groupby_col),
                        "range": (
                            pl.col(num_col).max().over(groupby_col)
                            - pl.col(num_col).min().over(groupby_col)
                        ),
                    }

                    new_columns.append(agg_functions[fun].alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
