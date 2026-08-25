"""Rolling-window statistics feature generator."""

from __future__ import annotations

import polars as pl
from pydantic import PositiveInt, PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

_VALID_FUNCS = frozenset({"mean", "std", "min", "max", "sum"})

_NUMERIC_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


def _rolling_expr(
    col_expr: pl.Expr,
    func: str,
    window_size: int,
    min_samples: int,
) -> pl.Expr:
    """Return the rolling expression for the given function name."""
    if func == "mean":
        return col_expr.rolling_mean(window_size=window_size, min_samples=min_samples)
    if func == "std":
        return col_expr.rolling_std(window_size=window_size, min_samples=min_samples)
    if func == "min":
        return col_expr.rolling_min(window_size=window_size, min_samples=min_samples)
    if func == "max":
        return col_expr.rolling_max(window_size=window_size, min_samples=min_samples)
    # sum
    return col_expr.rolling_sum(window_size=window_size, min_samples=min_samples)


class RollingStatisticsFeatures(_BaseTransformer):
    """Compute rolling window statistics for numeric columns.

    For each column in ``subset`` and each function in ``func`` a new feature
    column is appended with the naming pattern
    ``'{col}__rolling_{func}_{window_size}'``.

    When ``by`` is provided the rolling computation is performed
    independently within each group (rows already present in that group
    define the window order).

    Parameters
    ----------
    subset : list[str]
        Numeric columns to apply rolling statistics to.
    window_size : PositiveInt
        Number of rows in the rolling window.
    func : list[str]
        One or more statistics to compute.  Valid values:
        ``'mean'``, ``'std'``, ``'min'``, ``'max'``, ``'sum'``.
    by : list[str] or None, default=None
        Optional grouping columns.  When set, each statistic is computed
        independently within each group using Polars' ``over()`` expression.
        Ensure the DataFrame is sorted by group (and time, if applicable)
        before calling ``transform``.
    min_periods : PositiveInt or None, default=None
        Minimum number of non-null observations required to produce a value.
        Defaults to ``window_size`` when ``None`` (standard behaviour).
    drop_columns : bool, default=False
        When ``True``, drop the original ``subset`` columns after appending
        the new features.

    Attributes
    ----------
    new_column_names_ : list[str]
        Names of the generated feature columns (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation import RollingStatisticsFeatures

    >>> X = pl.DataFrame({
    ...     "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "count":  [10,  20,  30,  40,  50],
    ... })
    >>> transformer = RollingStatisticsFeatures(
    ...     subset=["amount", "count"],
    ...     window_size=3,
    ...     func=["mean", "sum"],
    ... )
    >>> transformer.fit(X)
    RollingStatisticsFeatures(subset=['amount', 'count'], window_size=3, func=['mean', 'sum'], by=None, min_periods=None, drop_columns=False)
    >>> transformer.transform(X)
    shape: (5, 6)
    ┌────────┬───────┬──────────────────────────┬───────────────────────────┬──────────────────────────┬───────────────────────────┐
    │ amount ┆ count ┆ amount__rolling_mean_3   ┆ amount__rolling_sum_3    ┆ count__rolling_mean_3    ┆ count__rolling_sum_3      │
    │ ---    ┆ ---   ┆ ---                      ┆ ---                      ┆ ---                      ┆ ---                       │
    │ f64    ┆ i64   ┆ f64                      ┆ f64                      ┆ f64                      ┆ i64                       │
    ╞════════╪═══════╪══════════════════════════╪══════════════════════════╪══════════════════════════╪═══════════════════════════╡
    │ 1.0    ┆ 10    ┆ null                     ┆ null                     ┆ null                     ┆ null                      │
    │ 2.0    ┆ 20    ┆ null                     ┆ null                     ┆ null                     ┆ null                      │
    │ 3.0    ┆ 30    ┆ 2.0                      ┆ 6.0                      ┆ 20.0                     ┆ 60                        │
    │ 4.0    ┆ 40    ┆ 3.0                      ┆ 9.0                      ┆ 30.0                     ┆ 90                        │
    │ 5.0    ┆ 50    ┆ 4.0                      ┆ 12.0                     ┆ 40.0                     ┆ 120                       │
    └────────┴───────┴──────────────────────────┴──────────────────────────┴──────────────────────────┴───────────────────────────┘
    """

    subset: list[str]
    window_size: PositiveInt
    func: list[str]
    by: list[str] | None = None
    min_periods: PositiveInt | None = None
    drop_columns: bool = False

    _new_column_names: list[str] = PrivateAttr(default_factory=list)

    @field_validator("func")
    @classmethod
    def _validate_func(cls, v: list[str]) -> list[str]:
        invalid = set(v) - _VALID_FUNCS
        if invalid:
            raise ValueError(
                f"Invalid function(s): {invalid}.  " f"Valid options are: {sorted(_VALID_FUNCS)}."
            )
        return v

    @property
    def new_column_names_(self) -> list[str]:
        """Names of the generated feature columns."""
        return self._new_column_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> RollingStatisticsFeatures:
        """Record the names of the output feature columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame (used only to confirm ``subset`` columns exist).
        y : pl.Series or None, default=None
            Ignored; present for sklearn compatibility.

        Returns
        -------
        RollingStatisticsFeatures
            The fitted transformer instance.
        """
        self._new_column_names = [
            f"{col}__rolling_{fn}_{self.window_size}" for col in self.subset for fn in self.func
        ]
        self._output_dtypes = {col: pl.Float64 for col in self._new_column_names}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Append rolling statistics columns to *X*.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with rolling feature columns appended.
        """
        min_samples = self.min_periods if self.min_periods is not None else self.window_size

        transformations = []
        for col in self.subset:
            for fn in self.func:
                new_col = f"{col}__rolling_{fn}_{self.window_size}"
                base = _rolling_expr(pl.col(col), fn, self.window_size, min_samples)
                if self.by:
                    expr = base.over(self.by).alias(new_col)
                else:
                    expr = base.alias(new_col)
                transformations.append(expr)

        result = X.with_columns(transformations)
        if self.drop_columns:
            result = result.drop(self.subset)
        return result
