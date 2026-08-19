"""Fourier (sin/cos) features for cyclic numeric columns."""

from __future__ import annotations

import math

import polars as pl
from pydantic import PositiveInt, PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer


class FourierFeatures(_BaseTransformer):
    """Append sine and cosine Fourier features to numeric columns.

    For each column *c* in ``subset``, each period *T* in the column's
    period list, and each harmonic *k* from 1 to ``n_harmonics``, two new
    columns are created:

    .. math::

        c\\_\\text{fourier\\_sin}\\_p{T}\\_k{k} &= \\sin\\!\\left(\\frac{2\\pi k \\cdot c}{T}\\right)

        c\\_\\text{fourier\\_cos}\\_p{T}\\_k{k} &= \\cos\\!\\left(\\frac{2\\pi k \\cdot c}{T}\\right)

    Fourier features are useful for encoding cyclic patterns when the period
    is known (e.g. day-of-week with T=7, month with T=12, or arbitrary
    numerical cycles).  Unlike simple trigonometric encoding, using multiple
    harmonics captures non-sinusoidal shapes in the cycle.

    Parameters
    ----------
    subset : list[str]
        Numeric columns to transform.
    periods : list[float] | dict[str, list[float]]
        Periods to apply.

        - **list[float]**: the same list of periods is applied to *every*
          column in ``subset``.
        - **dict[str, list[float]]**: maps each column name to its own list
          of periods.  Every column in ``subset`` must appear as a key.
    n_harmonics : PositiveInt, default=1
        Number of harmonics per period.  ``n_harmonics=1`` gives only the
        fundamental frequency; ``n_harmonics=2`` adds the first overtone, etc.
    drop_columns : bool, default=False
        Drop the original ``subset`` columns after appending Fourier features.

    Attributes
    ----------
    column_periods_ : dict[str, list[float]]
        Resolved per-column period mapping (set after ``fit``).

    Examples
    --------
    >>> import math
    >>> import polars as pl
    >>> from gators.feature_generation import FourierFeatures

    >>> X = pl.DataFrame({"day": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    >>> transformer = FourierFeatures(
    ...     subset=["day"],
    ...     periods=[7.0],
    ...     n_harmonics=1,
    ... )
    >>> transformer.fit(X)
    FourierFeatures(subset=['day'], periods=[7.0], n_harmonics=1, drop_columns=False)
    >>> result = transformer.transform(X)
    >>> result.columns
    ['day', 'day__fourier_sin_p7_k1', 'day__fourier_cos_p7_k1']

    With per-column periods:

    >>> transformer2 = FourierFeatures(
    ...     subset=["day"],
    ...     periods={"day": [7.0, 30.0]},
    ...     n_harmonics=2,
    ... )
    >>> transformer2.fit(X)
    FourierFeatures(subset=['day'], periods={'day': [7.0, 30.0]}, n_harmonics=2, drop_columns=False)
    >>> result2 = transformer2.transform(X)
    >>> result2.columns
    ['day', 'day__fourier_sin_p7_k1', 'day__fourier_cos_p7_k1',
     'day__fourier_sin_p7_k2', 'day__fourier_cos_p7_k2',
     'day__fourier_sin_p30_k1', 'day__fourier_cos_p30_k1',
     'day__fourier_sin_p30_k2', 'day__fourier_cos_p30_k2']
    """

    subset: list[str]
    periods: list[float] | dict[str, list[float]]
    n_harmonics: PositiveInt = 1
    drop_columns: bool = False

    _column_periods: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    @field_validator("periods")
    @classmethod
    def _validate_periods(
        cls, v: list[float] | dict[str, list[float]]
    ) -> list[float] | dict[str, list[float]]:
        if isinstance(v, list):
            if not v:
                raise ValueError("periods must not be empty.")
            for p in v:
                if p <= 0:
                    raise ValueError(f"All periods must be positive; got {p}.")
        else:
            for col, ps in v.items():
                if not ps:
                    raise ValueError(f"periods for column '{col}' must not be empty.")
                for p in ps:
                    if p <= 0:
                        raise ValueError(
                            f"All periods for column '{col}' must be positive; got {p}."
                        )
        return v

    @property
    def column_periods_(self) -> dict[str, list[float]]:
        """Resolved per-column period mapping."""
        return self._column_periods

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "FourierFeatures":
        """Resolve per-column periods from the ``periods`` parameter.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame (not used for computation).
        y : pl.Series or None, default=None
            Ignored; present for sklearn compatibility.

        Returns
        -------
        FourierFeatures
            The fitted transformer instance.
        """
        if isinstance(self.periods, list):
            self._column_periods = {col: list(self.periods) for col in self.subset}
        else:
            self._column_periods = {col: list(self.periods[col]) for col in self.subset}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Append Fourier feature columns to *X*.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with sin/cos Fourier columns appended.
        """
        two_pi = 2.0 * math.pi
        transformations = []

        for col in self.subset:
            col_expr = pl.col(col).cast(pl.Float64)
            for period in self._column_periods[col]:
                period_str = f"{period:g}"
                for k in range(1, self.n_harmonics + 1):
                    factor = two_pi * k / period
                    sin_col = f"{col}__fourier_sin_p{period_str}_k{k}"
                    cos_col = f"{col}__fourier_cos_p{period_str}_k{k}"
                    transformations.append((col_expr * factor).sin().alias(sin_col))
                    transformations.append((col_expr * factor).cos().alias(cos_col))

        result = X.with_columns(transformations)
        if self.drop_columns:
            result = result.drop(self.subset)
        return result
