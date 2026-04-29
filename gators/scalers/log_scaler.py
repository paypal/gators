from typing import Dict, List, Literal, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class LogScaler(_BaseTransformer):
    """
    Applies logarithm transformation with choice of base.

    Log transformation is useful for:

    - Reducing right skewness in data
    - Stabilizing variance
    - Converting multiplicative relationships to additive
    - Compressing large value ranges

    Supports three bases:

    - 'e': Natural logarithm ln(X) / log_e(X)
    - '10': Base-10 logarithm
    - '2': Base-2 logarithm

    Note: Only positive values can be transformed. Zero and negative values
    will result in null/inf values.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to transform. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    base : Literal['e', '10', '2'], default='e'
        The logarithm base to use:
        - 'e': ln(X)
        - '10': log10(X)
        - '2': log2(X)
    drop_columns : bool, default=True
        If True, drop the original columns after transformation.
        If False, keep both original and transformed columns.

    Examples
    --------
    Create an instance of the LogScaler class with natural log:

    >>> import polars as pl
    >>> from gators.scalers import LogScaler
    >>> scaler = LogScaler(subset=["sales", "revenue"], base="e")

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
    ┌───────────────┬──────────────────┐
    │ sales__log_ln ┆ revenue__log_ln  │
    │ ---           ┆ ---              │
    │ f64           ┆ f64              │
    ├───────────────┼──────────────────┤
    │ 0.0           ┆ 2.303            │
    │ 2.303         ┆ 4.605            │
    │ 4.605         ┆ 6.908            │
    │ 6.908         ┆ 9.210            │
    └───────────────┴──────────────────┘

    >>> # Using log10
    >>> scaler10 = LogScaler(subset=["count"], base="10")
    >>> X2 = pl.DataFrame({"count": [1, 10, 100, 1000]})
    >>> scaler10.fit(X2)
    >>> scaler10.transform(X2)
    shape: (4, 1)
    ┌─────────────────┐
    │ count__log_10   │
    │ ---             │
    │ f64             │
    ├─────────────────┤
    │ 0.0             │
    │ 1.0             │
    │ 2.0             │
    │ 3.0             │
    └─────────────────┘

    >>> # Using log2
    >>> scaler2 = LogScaler(subset=["size"], base="2")
    >>> X3 = pl.DataFrame({"size": [1, 2, 4, 8, 16]})
    >>> scaler2.fit(X3)
    >>> scaler2.transform(X3)
    shape: (5, 1)
    ┌──────────────┐
    │ size__log_2  │
    │ ---          │
    │ f64          │
    ├──────────────┤
    │ 0.0          │
    │ 1.0          │
    │ 2.0          │
    │ 3.0          │
    │ 4.0          │
    └──────────────┘
    """

    subset: Optional[List[str]] = None
    base: Literal["e", "10", "2"] = "e"
    _column_mapping: Dict[str, str]
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "LogScaler":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        LogScaler
            The fitted transformer instance.
        """
        if not self.subset:
            # Use set for O(1) dtype lookup instead of list O(n) lookup
            numeric_dtypes = {pl.Float64, pl.Int64, pl.Float32, pl.Int32}
            self.subset = [
                col
                for col, dtype in zip(X.columns, X.dtypes)
                if dtype in numeric_dtypes
            ]

        # Create suffix based on base
        if self.base == "e":
            suffix = "log_ln"
        elif self.base == "10":
            suffix = "log_10"
        else:  # '2'
            suffix = "log_2"

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
        # Pre-select log function once (avoid repeated conditionals in loop)
        if self.base == "e":
            log_func = lambda col: pl.col(col).log()
        elif self.base == "10":
            log_func = lambda col: pl.col(col).log10()
        else:  # '2'
            log_func = lambda col: pl.col(col).log(base=2)
        
        # Build all transformations using pre-selected function
        transformations = [
            log_func(col).alias(new) 
            for col, new in self._column_mapping.items()
        ]

        X = X.with_columns(transformations)

        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
