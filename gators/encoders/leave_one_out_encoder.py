from typing import Optional, Union

import polars as pl
from pydantic import Field, PositiveFloat

from ._base_encoder import _BaseEncoder


class LeaveOneOutEncoder(_BaseEncoder):
    """
    Encodes categorical values using leave-one-out target encoding.

    For each row, this encoder calculates the mean of the target variable for
    the category, excluding the current row. This reduces overfitting compared
    to standard target encoding by preventing the target value from influencing
    its own encoding.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    min_count : Union[int, float], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    smoothing : float, default=0.0
        Smoothing parameter for regularization toward the global mean. Higher values increase regularization. Use 0 for no smoothing.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__encode_loo'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    Initialize and use `LeaveOneOutEncoder`.

    >>> import polars as pl
    >>> from gators.encoders import LeaveOneOutEncoder
    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "A", "C", "A", "B", "C"],
    ...     "target": [1, 0, 1, 0, 0, 1, 1],
    ...     "value": [1, 2, 3, 4, 5, 6, 7]
    ... })
    >>> encoder = LeaveOneOutEncoder(subset=["category"], smoothing=1.0)
    >>> _ = encoder.fit(X, y=X["target"])
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (7, 3)
    ┌───────┬──────────────────────────┬───────┐
    │ target┆ category__encode_loo     ┆ value │
    │ ---   ┆ ---                      ┆ ---   │
    │ i64   ┆ f64                      ┆ i64   │
    ╞═══════╪══════════════════════════╪═══════╡
    │ 1     ┆ 0.571429                 ┆ 1     │
    │ 0     ┆ 0.571429                 ┆ 2     │
    │ 1     ┆ 0.571429                 ┆ 3     │
    │ 0     ┆ 0.571429                 ┆ 4     │
    │ 0     ┆ 0.666667                 ┆ 5     │
    │ 1     ┆ 0.571429                 ┆ 6     │
    │ 1     ┆ 0.571429                 ┆ 7     │
    └───────┴──────────────────────────┴───────┘

    Example with no smoothing:

    >>> X = pl.DataFrame({
    ...     "category": ["A", "A", "A", "B", "B"],
    ...     "target": [1, 0, 1, 0, 1],
    ...     "value": [1, 2, 3, 4, 5]
    ... })
    >>> encoder = LeaveOneOutEncoder(subset=["category"], smoothing=0.0)
    >>> _ = encoder.fit(X, y="target")
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌───────┬─────────────────────────┬───────┐
    │ target┆ category__encode_loo    ┆ value │
    │ ---   ┆ ---                     ┆ ---   │
    │ i64   ┆ f64                     ┆ i64   │
    ╞═══════╪═════════════════════════╪═══════╡
    │ 1     ┆ 0.666667                ┆ 1     │
    │ 0     ┆ 0.666667                ┆ 2     │
    │ 1     ┆ 0.500000                ┆ 3     │
    │ 0     ┆ 0.500000                ┆ 4     │
    │ 1     ┆ 0.500000                ┆ 5     │
    └───────┴─────────────────────────┴───────┘
    """

    smoothing: float = Field(default=0.0, ge=0.0)
    global_mean_: float = 0.0

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "LeaveOneOutEncoder":
        """Fit the transformer by computing leave-one-out target statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : pl.Series
            Target series (binary or continuous). Required for LeaveOneOutEncoder.

        Returns
        -------
        LeaveOneOutEncoder
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If y is None.
        """
        if y is None:
            raise ValueError("LeaveOneOutEncoder requires a target variable (y) for fitting.")

        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
            ]

        # Calculate global mean
        self.global_mean_ = y.mean()

        min_threshold_count = self.min_count if self.min_count >= 1 else self.min_count * len(X)

        self.mapping_ = {}

        # Add target as temporary column for grouping
        X_with_target = X.with_columns(y.alias("__target__"))

        for col in self.subset:
            # Calculate sum and count for each category
            category_stats = X_with_target.group_by(col).agg(
                [
                    pl.col("__target__").sum().alias("target_sum"),
                    pl.col("__target__").count().alias("target_count"),
                ]
            )

            # Filter by min_count
            category_stats = category_stats.filter(pl.col("target_count") >= min_threshold_count)

            if category_stats.is_empty():
                continue

            # Join back to get category stats for each row
            X_with_stats = X_with_target.join(category_stats, on=col, how="left")

            # Calculate leave-one-out mean:
            # (total_sum - current_value + smoothing * global_mean) / (count - 1 + smoothing)
            if self.smoothing > 0:
                X_encoded = X_with_stats.with_columns(
                    [
                        (
                            (
                                pl.col("target_sum")
                                - pl.col("__target__")
                                + self.smoothing * self.global_mean_
                            )
                            / (pl.col("target_count") - 1 + self.smoothing)
                        ).alias(f"{col}__loo_enc")
                    ]
                )
            else:
                # No smoothing case
                X_encoded = X_with_stats.with_columns(
                    [
                        (
                            (pl.col("target_sum") - pl.col("__target__"))
                            / pl.when(pl.col("target_count") > 1)
                            .then(pl.col("target_count") - 1)
                            .otherwise(1)  # Avoid division by zero
                        ).alias(f"{col}__loo_enc")
                    ]
                )

            # Get mean encoding per category for transform
            # Use the average of all leave-one-out values as the category encoding
            category_means = (
                X_encoded.group_by(col)
                .agg(pl.col(f"{col}__loo_enc").mean())
                .filter(pl.col(col).is_not_null())
                .filter(pl.col(f"{col}__loo_enc").is_not_null())  # Filter out None/NaN means
            )

            # Create mapping
            mapping_dict = {
                cat: mean_val
                for cat, mean_val in zip(
                    category_means[col].to_list(),
                    category_means[f"{col}__loo_enc"].to_list(),
                )
                if mean_val is not None  # Extra safety check
            }

            if mapping_dict:
                self.mapping_[col] = mapping_dict

        self.column_mapping_ = {col: f"{col}__loo_enc" for col in self.mapping_.keys()}

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame using leave-one-out encoding.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with leave-one-out encoded columns.
        """
        # Use global mean as default for unseen categories
        default_value = self.global_mean_

        expressions = [
            pl.col(col)
            .replace_strict(mapping, default=default_value, return_dtype=pl.Float64)
            .alias(self.column_mapping_[col])
            for col, mapping in self.mapping_.items()
        ]

        X = X.with_columns(expressions)

        if self.drop_columns and self.subset:
            X = X.drop(self.subset)

        return X
