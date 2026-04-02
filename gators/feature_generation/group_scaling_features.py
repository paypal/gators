from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel, PrivateAttr, field_validator
from sklearn.base import BaseEstimator, TransformerMixin

SCALING_FUNCTIONS = ["mean", "median", "zscore", "minmax"]


class GroupScalingFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates group-based scaling features for numerical columns.

    This transformer creates features like:
    
    - value / group_mean (most common: relative position vs average)
    - value / group_median (robust to outliers)
    - (value - group_mean) / group_std (z-score: standardized deviation)
    - (value - group_min) / (group_max - group_min) (min-max: 0-1 normalization)

    Importance for Fraud Detection
    -------------------------------
    Group scaling features are particularly valuable in fraud detection because they capture
    relative deviations from group-level behavior patterns. Fraudulent transactions often
    exhibit unusual characteristics compared to the typical behavior within their segments.
    
    - **mean/median ratios**: Show multiplicative deviation (e.g., 10x the group average)
    - **zscore**: Quantifies how many standard deviations away from group mean (e.g., 3σ anomaly)
    - **minmax**: Shows relative position within observed range (0=min, 1=max, handles negatives)
    
    These features are especially powerful when combined with various grouping dimensions
    (e.g., by merchant, customer segment, time of day, or geographic location) to capture
    different aspects of abnormal behavior.

    Parameters
    ----------
    subset : List[str]
        List of numerical column names to transform.
    by : List[str]
        List of column names to use for groupby operations. Each column will be used
        for a separate groupby operation (e.g., ['cat1', 'cat2'] creates features
        grouped by cat1 and separate features grouped by cat2).
    func : List[str]
        List of scaling functions to apply. Available options:
        - 'mean': value / group_mean (relative position vs average)
        - 'median': value / group_median (robust to outliers)
        - 'zscore': (value - group_mean) / group_std (standardized deviation)
        - 'minmax': (value - group_min) / (group_max - group_min) (0-1 normalization)
    fill_value : float, default=0.0
        Value to use when denominator is zero or null (safe division/scaling).
    drop_columns : bool, default=False
        Whether to drop the original numerical columns after creating scaled features.
    new_column_names : Optional[List[str]], default=None
        List of custom names for the scaled feature columns. If None, uses default naming pattern
        '{num_col}__{func}_{groupby_col}'. Must have same length as the total number
        of features created (subset × by × func).

    Examples
    --------
    >>> from gators.feature_generation import GroupScalingFeatures
    >>> import polars as pl

    >>> X ={
    ...     'amount': [100, 200, 150, 300, 250],
    ...     'cat1': ['A', 'A', 'B', 'B', 'A'],
    ...     'cat2': ['X', 'Y', 'X', 'X', 'X']
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Single groupby column with multiple scaling functions**

    >>> transformer = GroupScalingFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['mean', 'zscore']
    ... )
    >>> transformer.fit(X)
    GroupScalingFeatures(subset=['amount'], by=['cat1'], func=['mean', 'zscore'])
    >>> result = transformer.transform(X)
    >>> result
    shape: (5, 5)
    ┌────────┬──────┬──────┬──────────────────┬────────────────────┐
    │ amount ┆ cat1 ┆ cat2 ┆ amount__mean_cat1 ┆ amount__zscore_cat1 │
    │ ---    ┆ ---  ┆ ---  ┆ ---              ┆ ---                │
    │ i64    ┆ str  ┆ str  ┆ f64              ┆ f64                │
    ╞════════╪══════╪══════╪══════════════════╪════════════════════╡
    │ 100    ┆ A    ┆ X    ┆ 0.545455         ┆ -1.069045          │
    │ 200    ┆ A    ┆ Y    ┆ 1.090909         ┆ 0.267261           │
    │ 150    ┆ B    ┆ X    ┆ 0.666667         ┆ -0.707107          │
    │ 300    ┆ B    ┆ X    ┆ 1.333333         ┆ 0.707107           │
    │ 250    ┆ A    ┆ X    ┆ 1.363636         ┆ 0.801784           │
    └────────┴──────┴──────┴──────────────────┴────────────────────┘

    **Example 2: Multiple groupby columns**

    >>> X ={
    ...     'amount': [100, 200, 150, 300],
    ...     'value': [50, 100, 75, 150],
    ...     'cat1': ['A', 'A', 'B', 'B'],
    ...     'cat2': ['X', 'Y', 'X', 'Y']
    ... }
    >>> X = pl.DataFrame(X)
    >>> transformer = GroupScalingFeatures(
    ...     subset=['amount'],
    ...     by=['cat1', 'cat2'],
    ...     func=['mean']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['amount', 'value', 'cat1', 'cat2', 'amount__mean_cat1', 'amount__mean_cat2']
    # Creates separate features grouped by cat1 and grouped by cat2

    **Example 3: Min-max scaling**

    >>> X ={
    ...     'amount': [100, 200, 150, 300],
    ...     'cat1': ['A', 'A', 'B', 'B']
    ... }
    >>> X = pl.DataFrame(X)
    >>> transformer = GroupScalingFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['minmax']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 3)
    ┌────────┬──────┬─────────────────────┐
    │ amount ┆ cat1 ┆ amount__minmax_cat1 │
    │ ---    ┆ ---  ┆ ---                 │
    │ i64    ┆ str  ┆ f64                 │
    ╞════════╪══════╪═════════════════════╡
    │ 100    ┆ A    ┆ 0.0                 │
    │ 200    ┆ A    ┆ 1.0                 │
    │ 150    ┆ B    ┆ 0.0                 │
    │ 300    ┆ B    ┆ 1.0                 │
    └────────┴──────┴─────────────────────┘
    """

    subset: List[str]
    by: List[str]
    func: List[str]
    fill_value: float = 0.0
    drop_columns: bool = False
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("func")
    def check_func(cls, func):
        for fun in func:
            if fun not in SCALING_FUNCTIONS:
                raise ValueError(
                    f"{fun} is not in the predefined list of scaling functions: {SCALING_FUNCTIONS}"
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
                    f"must match the total number of features created ({expected_length})"
                )
        return new_column_names

    def fit(
        self, X: pl.DataFrame, y: Optional[pl.Series] = None
    ) -> "GroupScalingFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        GroupScalingFeatures
            Fitted transformer instance.
        """
        default_names = []
        for num_col in self.subset:
            for groupby_col in self.by:
                for fun in self.func:
                    default_names.append(f"{num_col}__{fun}_{groupby_col}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating group scaling features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with group scaling features.
        """
        new_columns = []

        for num_col in self.subset:
            for groupby_col in self.by:
                for fun in self.func:
                    default_name = f"{num_col}__{fun}_{groupby_col}"
                    new_col_name = self._column_mapping[default_name]

                    # Compute group statistics
                    # Note: .over(groupby_col) must be applied AFTER the aggregation function
                    if fun == "mean":
                        # value / group_mean
                        denominator = pl.col(num_col).mean().over(groupby_col)
                        scaling_expr = (
                            pl.when((denominator == 0) | denominator.is_null())
                            .then(self.fill_value)
                            .otherwise(pl.col(num_col) / denominator)
                            .alias(new_col_name)
                        )
                    elif fun == "median":
                        # value / group_median
                        denominator = pl.col(num_col).median().over(groupby_col)
                        scaling_expr = (
                            pl.when((denominator == 0) | denominator.is_null())
                            .then(self.fill_value)
                            .otherwise(pl.col(num_col) / denominator)
                            .alias(new_col_name)
                        )
                    elif fun == "zscore":
                        # (value - group_mean) / group_std
                        group_mean = pl.col(num_col).mean().over(groupby_col)
                        group_std = pl.col(num_col).std().over(groupby_col)
                        scaling_expr = (
                            pl.when((group_std == 0) | group_std.is_null())
                            .then(self.fill_value)
                            .otherwise((pl.col(num_col) - group_mean) / group_std)
                            .alias(new_col_name)
                        )
                    elif fun == "minmax":
                        # (value - group_min) / (group_max - group_min)
                        group_min = pl.col(num_col).min().over(groupby_col)
                        group_max = pl.col(num_col).max().over(groupby_col)
                        range_val = group_max - group_min
                        scaling_expr = (
                            pl.when((range_val == 0) | range_val.is_null())
                            .then(self.fill_value)
                            .otherwise((pl.col(num_col) - group_min) / range_val)
                            .alias(new_col_name)
                        )

                    new_columns.append(scaling_expr)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
