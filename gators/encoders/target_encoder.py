import numpy as np
import polars as pl

from ._base_encoder import _BaseEncoder


class TargetEncoder(_BaseEncoder):
    """
    Target-based encoded categorical values.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    min_count : Union[int, float], default=1
        Minimum count threshold for encoding categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__target_enc'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    Basic usage:

    >>> from gators.encoders import TargetEncoder
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     "A": ["foo", "bar", "foo", "bar", "baz"],
    ...     "B": [True, False, True, True, False],
    ... })
    >>> target = pl.Series("target", [1, 0, 1, 1, 0])
    >>> encoder = TargetEncoder(inplace=False, drop_columns=True)
    >>> encoder.fit(X, target)
    TargetEncoder(...)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌───────────────┬───────────────┐
    │ B__target_enc ┆ A__target_enc │
    │ ---           ┆ ---           │
    │ f64           ┆ f64           │
    ╞═══════════════╪═══════════════╡
    │ 1.0           ┆ 1.0           │
    │ 0.0           ┆ 0.5           │
    │ 1.0           ┆ 1.0           │
    │ 1.0           ┆ 0.5           │
    │ 0.0           ┆ 0.0           │
    └───────────────┴───────────────┘

    Drop columns:

    >>> encoder = TargetEncoder(drop_columns=False, inplace=False)
    >>> encoder.fit(X, target)
    TargetEncoder(...)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────────────┬─────────────┬───────────────┬───────────────┐
    │ A           │ B           │ A__target_enc │ B__target_enc │
    │ str         │ bool        │ f64           │ f64           │
    ╞═════════════╪═════════════╪═══════════════╪═══════════════╡
    │ foo         │ true        │ 1.0           │ 1.0           │
    │ bar         │ false       │ 1.0           │ 0.0           │
    │ foo         │ true        │ 1.0           │ 1.0           │
    │ bar         │ true        │ 1.0           │ 1.0           │
    │ baz         │ false       │ 0.0           │ 0.0           │
    └─────────────┴─────────────┴───────────────┴─────────────┘

    Subset of columns:

    >>> encoder = TargetEncoder(subset=["A"], inplace=False, drop_columns=True)
    >>> encoder.fit(X, target)
    TargetEncoder(...)
    >>> transformed_X = encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 1)
    ┌───────────────┐
    │ A__target_enc │
    │ f64           │
    ╞═══════════════╡
    │ 1.0           │
    │ 1.0           │
    │ 1.0           │
    │ 1.0           │
    │ 0.0           │
    └───────────────┘

    """

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "TargetEncoder":
        """Fit the transformer by computing target mean for each category.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : pl.Series
            Target series (binary or continuous).

        Returns
        -------
        TargetEncoder
            The fitted transformer instance.
        """
        # Validate that y is provided
        if y is None:
            raise ValueError(
                "TargetEncoder requires a target variable 'y' for fitting. "
                "Please provide y when calling fit() or fit_transform(), e.g., "
                "encoder.fit(X, y=y_train) or pipeline.fit_transform(X, y=y_train)"
            )
        
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
            ]
        # Add target as a temporary column for unpivoting
        X_with_target = X.select(self.subset).with_columns(y.alias("__target__"))
        melted = X_with_target.unpivot(index="__target__")
        stats = melted.group_by(["variable", "value"]).agg(
            [
                pl.col("__target__").mean().alias("mean"),
                pl.col("__target__").count().alias("N"),
            ]
        )
        self.mapping_ = {}
        min_count_threshold = (
            self.min_count if self.min_count >= 1 else len(X) * self.min_count
        )
        for col in set(stats["variable"]):
            stats_col = stats.filter(stats["variable"] == col)
            self.mapping_[col] = {
                cat: val
                for cat, val, count in stats_col[["value", "mean", "N"]].iter_rows()
                if count >= min_count_threshold
            }
        self.column_mapping_ = {col: f"{col}__target_enc" for col in self.subset}
        return self
