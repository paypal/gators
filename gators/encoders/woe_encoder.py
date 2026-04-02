from typing import Annotated, List, Optional, Union

import numpy as np
import polars as pl
from pydantic import Field, PositiveFloat, PositiveInt

from ._base_encoder import _BaseEncoder


def compute_woe_iv(
    X: pl.DataFrame,
    y: pl.Series,
    regularization: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = 0.01,
):
    # Pre-compute target statistics for better performance
    reg = regularization if regularization is not None else 0.01
    num_1s = y.sum()
    num_0s = len(y) - num_1s
    denom_1 = num_1s + 2 * reg
    denom_0 = num_0s + 2 * reg

    # Compute WOE and IV statistics in optimized chain
    stats = (
        X.with_columns(y.alias("__target__"))
        .unpivot(index="__target__")
        .group_by(["variable", "value"])
        .agg([
            pl.col("__target__").sum().alias("1"),
            pl.col("__target__").count().alias("N"),
        ])
        .with_columns([
            (pl.col("N") - pl.col("1")).alias("0"),
            ((pl.col("1") + reg) / denom_1).alias("distrib_1"),
            ((pl.col("N") - pl.col("1") + reg) / denom_0).alias("distrib_0"),
        ])
        .with_columns(
            (pl.col("distrib_1") / pl.col("distrib_0")).log().alias("woe")
        )
        .with_columns(
            ((pl.col("distrib_1") - pl.col("distrib_0")) * pl.col("woe")).alias("iv")
        )
        .sort("woe", descending=True)
    )
    return stats


class WOEEncoder(_BaseEncoder):
    """
    Weight of Evidence (WOE) encodes categorical variables.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of categorical columns to encode. If None, all string, boolean, and categorical columns are selected.
    regularization : Optional[float], default=0.01
        Regularization term (0.0-1.0) to prevent division by zero in WOE calculation.
    default : float, default=0.0
        Default WOE value for categories with insufficient counts or unseen categories.
    min_count : Union[PositiveInt, PositiveFloat], default=1
        Minimum count threshold for categories. If >= 1, treated as absolute count; if < 1, treated as frequency.
    inplace : bool, default=True
        If True, replace original columns with encoded values.
        If False, create new columns with suffix '__encode_woe'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after encoding.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.encoders import WOEEncoder

    >>> # Sample data
    >>> X = pl.DataFrame({
    ...     'A': ['cat', 'dog', 'cat', 'dog', 'cat'],
    ...     'B': ['x', 'x', 'y', 'y', 'x']
    ... })
    >>> y = pl.Series('target', [1, 0, 1, 1, 0])

    >>> encoder = WOEEncoder(inplace=False, drop_columns=True)
    >>> _ = encoder.fit(X, y)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌────────────────┬────────────────┐
    │ A__encode_woe  │ B__encode_woe  │
    │ ---            │ ---            │
    │ f64            │ f64            │
    ├────────────────┼────────────────┤
    │ 0.287682       │ -1.090344      │
    │ -0.402159      │ -1.090344      │
    │ 0.287682       │ 4.901146       │
    │ -0.402159      │ 4.901146       │
    │ 0.287682       │  -1.090344     │
    └────────────────┴────────────────┘

    >>> # Encoding with drop_columns=False
    >>> encoder = WOEEncoder(inplace=False, inplace=False, drop_columns=False)
    >>> encoder.fit(X, y)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 4)
    ┌─────┬─────┬────────────────┬────────────────┐
    │ A   │ B   │ A__encode_woe  │ B__encode_woe  │
    │ --- │ --- │ ---            │ ---            │
    │ str │ str │ f64            │ f64            │
    ├─────┼─────┼────────────────┼────────────────┤
    │ cat │ x   │ 0.287682       │ 0.287682       │
    │ dog │ x   │ -1.203973      │ 0.287682       │
    │ cat │ y   │ 0.287682       │ -1.203973      │
    │ dog │ y   │ -1.203973      │ -1.203973      │
    │ cat │ x   │ 0.287682       │ 0.287682       │
    └─────┴─────┼────────────────┼────────────────┘

    >>> # Encoding with columns as a subset
    >>> encoder = WOEEncoder(subset=['A'], inplace=False, drop_columns=False)
    >>> encoder.fit(X, y)
    >>> transformed_X =encoder.transform(X)
    >>> print(transformed_X)
    shape: (5, 3)
    ┌─────┬───────┬───────────────┬───────────────┐
    │ A   ┆ B     ┆ B__target_enc ┆ A__target_enc │
    │ --- ┆ ---   ┆ ---           ┆ ---           │
    │ str ┆ bool  ┆ f64           ┆ f64           │
    ╞═════╪═══════╪═══════════════╪═══════════════╡
    │ foo ┆ true  ┆ 1.0           ┆ 1.0           │
    │ bar ┆ false ┆ 0.0           ┆ 0.5           │
    │ foo ┆ true  ┆ 1.0           ┆ 1.0           │
    │ bar ┆ true  ┆ 1.0           ┆ 0.5           │
    │ baz ┆ false ┆ 0.0           ┆ 0.0           │
    └─────┴───────┴───────────────┴───────────────┘
    """

    subset: Optional[List[str]] = None
    regularization: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = 0.01
    default: float = 0.0
    min_count: Union[PositiveInt, PositiveFloat] = 1
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "WOEEncoder":
        """Fit the transformer by computing Weight of Evidence values for each category.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with categorical columns.
        y : pl.Series
            Binary target series (must contain 0s and 1s).

        Returns
        -------
        WOEEncoder
            The fitted transformer instance.
        """
        # Validate that y is provided
        if y is None:
            raise ValueError(
                "WOEEncoder requires a target variable 'y' for fitting. "
                "Please provide y when calling fit() or fit_transform(), e.g., "
                "encoder.fit(X, y=y_train) or pipeline.fit_transform(X, y=y_train)"
            )
        
        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in [pl.String, pl.Boolean, pl.Categorical]
            ]
        X = X.with_columns([pl.col(col).fill_null("MISSING_") for col in self.subset])
        stats = compute_woe_iv(
            X=X.select(self.subset),
            y=y,
            regularization=self.regularization,
        )
        self.mapping_ = {}
        min_count_threshold = (
            self.min_count if self.min_count >= 1 else len(X) * self.min_count
        )
        for key, group_df in stats.group_by("variable"):
            col = key[0] if isinstance(key, tuple) else key
            self.mapping_[col] = {
                str(cat): float(val)
                for cat, val, count in group_df[["value", "woe", "N"]].iter_rows()
                if count >= min_count_threshold
            }
        self.column_mapping_ = {col: f"{col}__encode_woe" for col in self.subset}
        return self
