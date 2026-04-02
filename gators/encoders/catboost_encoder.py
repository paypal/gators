from typing import Optional, Union

import polars as pl
from pydantic import Field, PositiveFloat

from ._base_encoder import _BaseEncoder


class CatBoostEncoder(_BaseEncoder):
    """
    Encodes categorical values using CatBoost target encoding with ordered statistics.

    This encoder implements the CatBoost algorithm's approach to target encoding,
    which uses ordered target statistics to prevent target leakage and overfitting.
    For each category, it calculates the cumulative mean of the target up to (but
    not including) the current row.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    min_count : Union[int, float], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    smoothing : float, default=1.0
        Smoothing parameter for regularization toward the global mean. Higher values increase regularization.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__encode_catboost'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    Initialize and use `CatBoostEncoder`.

    >>> import polars as pl
    >>> from gators.encoders import CatBoostEncoder
    >>> X = pl.DataFrame({
    ...     "category": ["A", "B", "A", "C", "A", "B", "C"],
    ...     "value": [1, 2, 3, 4, 5, 6, 7]
    ... })
    >>> y = pl.Series("target", [1, 0, 1, 0, 0, 1, 1])
    >>> encoder = CatBoostEncoder(subset=["category"], smoothing=1.0, inplace=False, drop_columns=True)
    >>> _ = encoder.fit(X, y)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (7, 3)
    ┌───────┬────────────────────────────┬───────┐
    │ target┆ category__encode_catboost  │ value │
    │ ---   ┆ ---                        ┆ ---   │
    │ i64   ┆ f64                        ┆ i64   │
    ╞═══════╪════════════════════════════╪═══════╡
    │ 1     ┆ 0.571429                   ┆ 1     │
    │ 0     ┆ 0.571429                   ┆ 2     │
    │ 1     ┆ 0.666667                   ┆ 3     │
    │ 0     ┆ 0.571429                   ┆ 4     │
    │ 0     ┆ 0.600000                   ┆ 5     │
    │ 1     ┆ 0.428571                   ┆ 6     │
    │ 1     ┆ 0.428571                   ┆ 7     │
    └───────┴────────────────────────────┴───────┘
    """

    smoothing: PositiveFloat = Field(default=1.0)
    global_mean_: float = 0.0

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CatBoostEncoder":
        """Fit the transformer by computing CatBoost ordered target statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : Optional[pl.Series], default=None
            Target series (binary or continuous). Required for CatBoostEncoder.

        Returns
        -------
        CatBoostEncoder
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If y is None.
        """
        if y is None:
            raise ValueError("CatBoostEncoder requires a target variable 'y'")

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

        # Add target as temporary column
        X_with_target = X.with_columns(y.alias("__target__"))

        for col in self.subset:
            X_indexed = X_with_target.with_row_index("__row_idx")

            cumsum_expr = (
                X_indexed.sort("__row_idx")
                .group_by(col, maintain_order=True)
                .agg(
                    [
                        pl.col("__target__").cum_sum().alias("cumsum"),
                        pl.col("__target__").cum_count().alias("cumcount"),
                    ]
                )
                .explode(["cumsum", "cumcount"])
            )

            X_with_stats = X_indexed.join(cumsum_expr, on=col, how="left").sort("__row_idx")

            # Calculate encoding: (cumsum - current_value + smoothing * global_mean) / (cumcount - 1 + smoothing)
            # This excludes the current row (ordered target statistic)
            X_encoded = X_with_stats.with_columns(
                [
                    (
                        (
                            pl.col("cumsum")
                            - pl.col("__target__")
                            + self.smoothing * self.global_mean_
                        )
                        / (pl.col("cumcount") - 1 + self.smoothing)
                    ).alias(f"{col}__encode_catboost")
                ]
            )

            # Get mean encoding per category for transform
            category_means = X_encoded.group_by(col).agg(pl.col(f"{col}__encode_catboost").mean())

            # Filter by min_count
            value_counts = X[col].value_counts()
            valid_categories = set(
                cat
                for cat, count in zip(value_counts[col].to_list(), value_counts["count"].to_list())
                if count >= min_threshold_count
            )

            # Create mapping
            mapping_dict = {
                cat: mean_val
                for cat, mean_val in zip(
                    category_means[col].to_list(),
                    category_means[f"{col}__encode_catboost"].to_list(),
                )
                if cat in valid_categories
            }

            if mapping_dict:
                self.mapping_[col] = mapping_dict

        self.column_mapping_ = {col: f"{col}__catboost_enc" for col in self.mapping_.keys()}

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame using CatBoost encoding.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with CatBoost encoded columns.
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
