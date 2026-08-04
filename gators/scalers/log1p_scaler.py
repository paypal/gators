from typing import Literal

import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class Log1pScaler(_BaseTransformer):
    """
    Applies log1p transformation log(1+x) with choice of base.

    Log1p transformation is useful for:

    - Reducing right skewness in data
    - Stabilizing variance
    - Converting multiplicative relationships to additive
    - Compressing large value ranges while handling zero values

    Supports three bases:

    - 'e': Natural log1p ln(1+X)
    - '10': Base-10 log1p log10(1+X)
    - '2': Base-2 log1p log2(1+X)

    Note: Values must be greater than -1. Values of -1 or below will
    result in null/inf values.

    Parameters
    ----------
    subset : list[str], default=None
        List of numeric column names to transform. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    base : Literal['e', '10', '2'], default='e'
        The logarithm base to use:
        - 'e': ln(1+X)
        - '10': log10(1+X)
        - '2': log2(1+X)
    drop_columns : bool, default=True
        If True, drop the original columns after transformation.
        If False, keep both original and transformed columns.

    Examples
    --------
    Create an instance of the Log1pScaler class with natural log:

    >>> import polars as pl
    >>> from gators.scalers import Log1pScaler
    >>> scaler = Log1pScaler(subset=["sales", "revenue"], base="e")

    Fit the transformer:

    >>> X = pl.DataFrame({
    ...     "sales": [1, 10, 100, 1000],
    ...     "revenue": [10, 100, 1000, 10000]
    ... })
    >>> scaler.fit(X)

    Transform the DataFrame:

    >>> transformed_X = scaler.transform(X)
    >>> print(transformed_X)
    shape: (4, 2)
    ┌─────────────────┬───────────────────┐
    │ sales__log1p ┆ revenue__log1p │
    │ ---             ┆ ---               │
    │ f64             ┆ f64               │
    ├─────────────────┼───────────────────┤
    │ 0.693           ┆ 2.398             │
    │ 2.398           ┆ 4.615             │
    │ 4.615           ┆ 6.909             │
    │ 6.909           ┆ 9.210             │
    └─────────────────┴───────────────────┘

    >>> # Using log10
    >>> scaler10 = Log1pScaler(subset=["count"], base="10")
    >>> X2 = pl.DataFrame({"count": [1, 10, 100, 1000]})
    >>> scaler10.fit(X2)
    >>> scaler10.transform(X2)
    shape: (4, 1)
    ┌─────────────────┐
    │ count__log1p_10 │
    │ ---             │
    │ f64             │
    ├─────────────────┤
    │ 0.301           │
    │ 1.041           │
    │ 2.004           │
    │ 3.000           │
    └─────────────────┘

    >>> # Using log2
    >>> scaler2 = Log1pScaler(subset=["size"], base="2")
    >>> X3 = pl.DataFrame({"size": [1, 2, 4, 8, 16]})
    >>> scaler2.fit(X3)
    >>> scaler2.transform(X3)
    shape: (5, 1)
    ┌───────────────┐
    │ size__log1p_2 │
    │ ---           │
    │ f64           │
    ├───────────────┤
    │ 1.000         │
    │ 1.585         │
    │ 2.322         │
    │ 3.170         │
    │ 4.087         │
    └───────────────┘
    """

    subset: list[str] | None = None
    base: Literal["e", "10", "2"] = "e"
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "Log1pScaler":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : pl.Series, default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        Log1pScaler
            The fitted transformer instance.
        """
        if not self.subset:
            # Use set for O(1) dtype lookup instead of list O(n) lookup
            numeric_dtypes = {pl.Float64, pl.Int64, pl.Float32, pl.Int32}
            self.subset = [
                col for col, dtype in zip(X.columns, X.dtypes) if dtype in numeric_dtypes
            ]

        # Create suffix based on base
        if self.base == "e":
            suffix = "log1p"
        elif self.base == "10":
            suffix = "log1p_10"
        else:  # '2'
            suffix = "log1p_2"

        self._column_mapping = {col: f"{col}__{suffix}" for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying logarithm.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform. Values should be positive.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with log-transformed columns.

        Notes
        -----
        Zero and negative values will result in null or -inf values.
        """
        # Pre-select log1p function once (avoid repeated conditionals in loop)
        if self.base == "e":
            log_func = lambda col: (pl.col(col) + 1).log()
        elif self.base == "10":
            log_func = lambda col: (pl.col(col) + 1).log(base=10)
        else:  # '2'
            log_func = lambda col: (pl.col(col) + 1).log(base=2)

        # Build all transformations using pre-selected function
        transformations = [log_func(col).alias(new) for col, new in self._column_mapping.items()]

        X = X.with_columns(transformations)

        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Reverse the log1p transformation.

        Parameters
        ----------
        X : pl.DataFrame
            DataFrame with log-transformed columns (output of ``transform``).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns restored to their original scale.
        """
        reverse_map = {v: k for k, v in self._column_mapping.items()}
        if self.base == "e":
            exprs = [
                (pl.col(new).exp() - 1).alias(orig)
                for new, orig in reverse_map.items()
                if new in X.columns
            ]
        elif self.base == "10":
            exprs = [
                (pl.lit(10.0) ** pl.col(new) - 1).alias(orig)
                for new, orig in reverse_map.items()
                if new in X.columns
            ]
        else:  # "2"
            exprs = [
                (pl.lit(2.0) ** pl.col(new) - 1).alias(orig)
                for new, orig in reverse_map.items()
                if new in X.columns
            ]
        X = X.with_columns(exprs)
        if self.drop_columns:
            return X.drop([c for c in reverse_map if c in X.columns])
        return X
