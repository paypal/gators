from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class ArcSinhScaler(_BaseTransformer):
    """
    Applies inverse hyperbolic sine (arcsinh) transformation.

    The transformation asinh(X) = log(X + sqrt(X^2 + 1)) is similar to log
    transformation but can handle zero and negative values. It's useful for
    stabilizing variance in data with both positive and negative values or
    wide dynamic range.

    Properties:

    - Defined for all real numbers (unlike log)
    - Approximately linear near zero
    - Approximately logarithmic for large abs(X)
    - Symmetric around zero: asinh(-X) = -asinh(X)

    This transformation is commonly used in:

    - Financial data with positive and negative returns
    - Data with extreme outliers
    - Variables spanning multiple orders of magnitude with zeros

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to transform. If None, all numeric columns
        (Float64, Int64, Float32, Int32) are automatically selected.
    drop_columns : bool, default=True
        If True, drop the original columns after transformation.
        If False, keep both original and transformed columns.

    Examples
    --------
    Create an instance of the ArcSinhScaler class:

    >>> import polars as pl
    >>> from gators.scalers import ArcSinhScaler
    >>> scaler = ArcSinhScaler(subset=["returns", "profit"])

    Fit the transformer:

    >>> X = pl.DataFrame({
    ...     "returns": [-100, -10, 0, 10, 100],
    ...     "profit": [-50, -5, 0, 5, 50]
    ... })
    >>> scaler.fit(X)

    Transform the DataFrame:

    >>> transformed_X = scaler.transform(X)
    >>> print(transformed_X)
    shape: (5, 2)
    ┌───────────────────┬─────────────────┐
    │ returns__arcsinh  ┆ profit__arcsinh │
    │ ---               ┆ ---             │
    │ f64               ┆ f64             │
    ├───────────────────┼─────────────────┤
    │ -5.298            ┆ -4.606          │
    │ -2.998            ┆ -2.312          │
    │ 0.0               ┆ 0.0             │
    │ 2.998             ┆ 2.312           │
    │ 5.298             ┆ 4.606           │
    └───────────────────┴─────────────────┘

    Notes
    -----
    The transformation is symmetric:
    - asinh(X) ≈ log(2X) for large positive X
    - asinh(X) ≈ X for X near 0
    - asinh(-X) = -asinh(X)
    """

    subset: Optional[List[str]] = None
    _column_mapping: Dict[str, str]
    drop_columns: bool = True

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "ArcSinhScaler":
        """Fit the transformer by storing column names.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to fit.
        y : Optional[pl.Series], default=None
            Target series (not used, present for sklearn compatibility).

        Returns
        -------
        ArcSinhScaler
            The fitted transformer instance.
        """
        if not self.subset:
            # Use set for O(1) dtype lookup instead of list O(n) lookup
            numeric_dtypes = {pl.Float64, pl.Int64, pl.Float32, pl.Int32}
            self.subset = [
                col for col, dtype in zip(X.columns, X.dtypes) if dtype in numeric_dtypes
            ]

        # Pre-build column mapping dictionary
        self._column_mapping = {col: f"{col}__arcsinh" for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by applying asinh(X).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with arcsinh-transformed columns.
        """
        transformations = [
            pl.col(col).arcsinh().alias(new) for col, new in self._column_mapping.items()
        ]

        X = X.with_columns(transformations)

        if self.drop_columns and self.subset is not None:
            return X.drop(self.subset)
        return X
