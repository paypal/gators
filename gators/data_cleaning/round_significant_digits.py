from typing import Annotated

import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class RoundSignificantDigits(_BaseTransformer):
    """Round selected numeric columns to a given number of significant figures.

    Significant-figure rounding preserves the most meaningful digits of a
    value regardless of its magnitude (e.g., with ``n_digits=3``:
    ``0.001234 → 0.00123``, ``1234.0 → 1230.0``, ``-9876.5 → -9880.0``).

    Parameters
    ----------
    n_digits : int
        Number of significant figures to keep. Must be >= 1.
    subset : list[str], default=None
        Columns to round. When ``None``, all numeric columns in the
        DataFrame are rounded automatically.
    inplace : bool, default=True
        If ``True``, the original columns are replaced in-place.
        If ``False``, new columns named ``{col}__round_{n_digits}sig`` are
        added alongside the originals.
    drop_columns : bool, default=True
        Relevant only when ``inplace=False``. If ``True``, the original
        columns are dropped after the new rounded columns are added.
        Ignored when ``inplace=True``.

    Examples
    --------
    **Example 1: Round all numeric columns in-place (default)**

    >>> import polars as pl
    >>> from gators.data_cleaning import RoundSignificantDigits
    >>> X = pl.DataFrame({
    ...     "a": [0.001234, 1234.0, -9876.5],
    ...     "b": [3.14159, 0.0, 9.9999],
    ...     "label": ["x", "y", "z"],
    ... })
    >>> transformer = RoundSignificantDigits(n_digits=3)
    >>> transformer.fit(X)
    RoundSignificantDigits(n_digits=3, subset=None, inplace=True, drop_columns=True)
    >>> print(transformer.transform(X))
    shape: (3, 3)
    ┌──────────┬───────┬───────┐
    │ a        ┆ b     ┆ label │
    │ ---      ┆ ---   ┆ ---   │
    │ f64      ┆ f64   ┆ str   │
    ╞══════════╪═══════╪═══════╡
    │ 0.00123  ┆ 3.14  ┆ x     │
    │ 1230.0   ┆ 0.0   ┆ y     │
    │ -9880.0  ┆ 10.0  ┆ z     │
    └──────────┴───────┴───────┘

    **Example 2: Add rounded columns without dropping originals**

    >>> transformer = RoundSignificantDigits(
    ...     n_digits=2, subset=["a"], inplace=False, drop_columns=False
    ... )
    >>> transformer.fit(X)
    RoundSignificantDigits(n_digits=2, subset=['a'], inplace=False, drop_columns=False)
    >>> print(transformer.transform(X))
    shape: (3, 4)
    ┌──────────┬───────┬───────┬───────────────┐
    │ a        ┆ b     ┆ label ┆ a__round_2sig │
    │ ---      ┆ ---   ┆ ---   ┆ ---           │
    │ f64      ┆ f64   ┆ str   ┆ f64           │
    ╞══════════╪═══════╪═══════╪═══════════════╡
    │ 0.001234 ┆ 3.14  ┆ x     ┆ 0.0012        │
    │ 1234.0   ┆ 0.0   ┆ y     ┆ 1200.0        │
    │ -9876.5  ┆ 10.0  ┆ z     ┆ -9900.0       │
    └──────────┴───────┴───────┴───────────────┘

    **Example 3: Add rounded columns and drop originals**

    >>> transformer = RoundSignificantDigits(
    ...     n_digits=2, subset=["a"], inplace=False, drop_columns=True
    ... )
    >>> transformer.fit(X)
    RoundSignificantDigits(n_digits=2, subset=['a'], inplace=False, drop_columns=True)
    >>> print(transformer.transform(X))
    shape: (3, 3)
    ┌───────┬───────┬───────────────┐
    │ b     ┆ label ┆ a__round_2sig │
    │ ---   ┆ ---   ┆ ---           │
    │ f64   ┆ str   ┆ f64           │
    ╞═══════╪═══════╪═══════════════╡
    │ 3.14  ┆ x     ┆ 0.0012        │
    │ 0.0   ┆ y     ┆ 1200.0        │
    │ 10.0  ┆ z     ┆ -9900.0       │
    └───────┴───────┴───────────────┘

    Notes
    -----
    - All numeric columns (including integers) are cast to ``Float64`` during
      the rounding computation; the output columns therefore have dtype
      ``Float64``.
    - Zero values are returned as ``0.0`` (log10(0) is undefined).
    - Null values propagate unchanged.
    """

    n_digits: Annotated[int, Field(ge=1)]
    subset: list[str] | None = None
    inplace: bool = True
    drop_columns: bool = True
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "RoundSignificantDigits":
        """Fit the transformer by recording which columns to round.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Ignored; present for sklearn compatibility.

        Returns
        -------
        RoundSignificantDigits
            Fitted transformer instance.
        """
        if self.subset is None:
            self.subset = [col for col in X.columns if X[col].dtype.is_numeric()]
        if not self.inplace:
            self._column_mapping = {col: [f"{col}__round_{self.n_digits}sig"] for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Round the selected columns to the configured number of significant figures.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with rounded columns.
        """
        columns: list[str] = self.subset or []

        def _sigfig_expr(col: str, alias: str) -> pl.Expr:
            float_col = pl.col(col).cast(pl.Float64)
            magnitude = float_col.abs().log(base=10).floor()
            factor = pl.lit(10.0).pow(pl.lit(float(self.n_digits - 1)) - magnitude)
            rounded = (float_col * factor).round(0) / factor
            return pl.when(float_col == 0.0).then(pl.lit(0.0)).otherwise(rounded).alias(alias)

        if self.inplace:
            return X.with_columns([_sigfig_expr(col, col) for col in columns])

        X = X.with_columns([_sigfig_expr(col, new) for col, [new] in self._column_mapping.items()])
        if self.drop_columns:
            return X.drop(columns)
        return X
