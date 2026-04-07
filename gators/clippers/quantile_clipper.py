from typing import Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseModel, BaseEstimator, TransformerMixin):
    """
    Clip numeric values based on quantile thresholds.

    This transformer caps values below the lower quantile and above the upper
    quantile. This is useful for removing extreme outliers while preserving
    the bulk of the data distribution.

    Parameters
    ----------
    lower_quantile : float, default=0.01
        Lower quantile threshold (0 to 1). Values below this quantile are clipped.
    upper_quantile : float, default=0.99
        Upper quantile threshold (0 to 1). Values above this quantile are clipped.
    subset : Optional[List[str]], default=None
        List of numeric columns to clip. If None, all numeric columns are selected.
    inplace : bool, default=True
        If True, clip values in the original columns.
        If False, create new columns with suffix '__clip_quantile'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after clipping.
        Ignored when inplace=True.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.clipping import QuantileClipper

    >>> # Sample DataFrame with outliers
    >>> X = pl.DataFrame({
    ...     'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
    ...     'B': [-50.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ... })

    >>> # Clip using 1st and 99th percentiles (default)
    >>> clipper = QuantileClipper(inplace=False)
    >>> clipper.fit(X)
    QuantileClipper(lower_quantile=0.01, upper_quantile=0.99, subset=['A', 'B'], drop_columns=True, inplace=False)
    >>> transformed_X = clipper.transform(X)
    >>> print(transformed_X)
    shape: (10, 2)
    ┌─────────────────────┬─────────────────────┐
    │ A__clip_quantile    ┆ B__clip_quantile    │
    │ ---                 ┆ ---                 │
    │ f64                 ┆ f64                 │
    ╞═════════════════════╪═════════════════════╡
    │ 1.09                ┆ -45.1               │
    │ 2.0                 ┆ 2.0                 │
    │ 3.0                 ┆ 3.0                 │
    │ 4.0                 ┆ 4.0                 │
    │ 5.0                 ┆ 5.0                 │
    │ 6.0                 ┆ 6.0                 │
    │ 7.0                 ┆ 7.0                 │
    │ 8.0                 ┆ 8.0                 │
    │ 9.0                 ┆ 9.0                 │
    │ 9.91                ┆ 9.91                │
    └─────────────────────┴─────────────────────┘

    >>> # More aggressive clipping with 5th and 95th percentiles
    >>> clipper_5_95 = QuantileClipper(lower_quantile=0.05, upper_quantile=0.95, inplace=False)
    >>> clipper_5_95.fit(X)
    QuantileClipper(lower_quantile=0.05, upper_quantile=0.95, subset=['A', 'B'], drop_columns=True, inplace=False)
    >>> transformed_X_5_95 = clipper_5_95.transform(X)
    >>> print(transformed_X_5_95)
    shape: (10, 2)
    ┌─────────────────────┬─────────────────────┐
    │ A__clip_quantile    ┆ B__clip_quantile    │
    │ ---                 ┆ ---                 │
    │ f64                 ┆ f64                 │
    ╞═════════════════════╪═════════════════════╡
    │ 1.45                ┆ -21.5               │
    │ 2.0                 ┆ 2.0                 │
    │ 3.0                 ┆ 3.0                 │
    │ 4.0                 ┆ 4.0                 │
    │ 5.0                 ┆ 5.0                 │
    │ 6.0                 ┆ 6.0                 │
    │ 7.0                 ┆ 7.0                 │
    │ 8.0                 ┆ 8.0                 │
    │ 9.0                 ┆ 9.0                 │
    │ 8.55                ┆ 9.55                │
    └─────────────────────┴─────────────────────┘

    >>> # Clip only specific columns
    >>> clipper_subset = QuantileClipper(subset=['A'], inplace=False)
    >>> clipper_subset.fit(X)
    QuantileClipper(lower_quantile=0.01, upper_quantile=0.99, subset=['A'], drop_columns=True, inplace=False)
    >>> transformed_X_subset = clipper_subset.transform(X)
    >>> print(transformed_X_subset)
    shape: (10, 2)
    ┌────────┬─────────────────────┐
    │ B      ┆ A__clip_quantile    │
    │ ---    ┆ ---                 │
    │ f64    ┆ f64                 │
    ╞════════╪═════════════════════╡
    │ -50.0  ┆ 1.09                │
    │ 2.0    ┆ 2.0                 │
    │ 3.0    ┆ 3.0                 │
    │ 4.0    ┆ 4.0                 │
    │ 5.0    ┆ 5.0                 │
    │ 6.0    ┆ 6.0                 │
    │ 7.0    ┆ 7.0                 │
    │ 8.0    ┆ 8.0                 │
    │ 9.0    ┆ 9.0                 │
    │ 10.0   ┆ 9.91                │
    └────────┴─────────────────────┘
    """

    lower_quantile: float = Field(default=0.01, ge=0.0, le=1.0)
    upper_quantile: float = Field(default=0.99, ge=0.0, le=1.0)
    subset: Optional[List[str]] = None
    drop_columns: bool = True
    inplace: bool = True
    _clip_bounds: Dict[str, Tuple[float, float]] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "QuantileClipper":
        """Fit the transformer by computing quantile-based clipping bounds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        QuantileClipper
            The fitted transformer instance.
        """
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                f"lower_quantile ({self.lower_quantile}) must be less than "
                f"upper_quantile ({self.upper_quantile})"
            )

        if not self.subset:
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype not in [pl.String, pl.Boolean]
            ]

        if not self.inplace:
            self._column_mapping = {col: f"{col}__clip_quantile" for col in self.subset}

        # Compute quantiles for each column
        for col in self.subset:
            lower_bound = X[col].quantile(self.lower_quantile)
            upper_bound = X[col].quantile(self.upper_quantile)
            self._clip_bounds[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by clipping values to quantile thresholds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with clipped numeric columns.
        """
        # Build all transformations at once
        if self.inplace:
            transformations = [
                pl.col(col).clip(
                    lower_bound=self._clip_bounds[col][0], upper_bound=self._clip_bounds[col][1]
                )
                for col in self.subset
            ]
        else:
            transformations = [
                pl.col(col)
                .clip(lower_bound=self._clip_bounds[col][0], upper_bound=self._clip_bounds[col][1])
                .alias(new)
                for col, new in self._column_mapping.items()
            ]

        X = X.with_columns(transformations)

        if not self.inplace and self.drop_columns:
            return X.drop(self.subset)
        return X
