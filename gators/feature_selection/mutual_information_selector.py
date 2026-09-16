"""Mutual information-based feature selector."""

from __future__ import annotations

import math

import polars as pl
from pydantic import PositiveInt, PrivateAttr

from ._base_selector import _BaseSelector

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


class MutualInformationSelector(_BaseSelector):
    """Drop columns whose Mutual Information with the target falls below a threshold.

    Mutual Information (MI) measures the statistical dependence between a
    feature and the target variable, without assuming a linear relationship.
    This makes it suitable for detecting non-linear associations that
    correlation-based selectors miss.

    For categorical (``String``, ``Boolean``) features, MI is computed directly
    from joint frequency counts.  For numeric features, values are first
    discretized into ``n_bins`` equal-width bins before counting.

    Parameters
    ----------
    threshold : float, default=0.0
        Minimum MI score required to keep a column.  Columns with MI strictly
        below this value are dropped.
    n_bins : PositiveInt, default=10
        Number of equal-width bins used to discretize numeric features.
        Higher values capture finer structure but require more data.
    subset : list[str] or None, default=None
        Columns to score.  If ``None``, every column in the DataFrame is
        evaluated.

    Attributes
    ----------
    selected_features_ : list[str]
        Column names that survive the threshold (set after ``fit``).
    columns_to_drop_ : list[str]
        Column names dropped because their MI was too low (set after ``fit``).
    mi_values_ : dict[str, float]
        Mapping of feature name to its computed MI score (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_selection import MutualInformationSelector

    >>> X = pl.DataFrame({
    ...     "informative": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ...     "noise":       [5.0, 1.0, 3.0, 2.0, 4.0, 6.0],
    ...     "category":   ["a", "b", "a", "b", "a", "b"],
    ... })
    >>> y = pl.Series("target", [0, 1, 0, 1, 0, 1])
    >>> selector = MutualInformationSelector(threshold=0.1)
    >>> selector.fit(X, y)
    MutualInformationSelector(threshold=0.1, n_bins=10, subset=None)
    >>> X_transformed = selector.transform(X)
    """

    threshold: float = 0.0
    n_bins: PositiveInt = 10
    subset: list[str] | None = None

    _mi_values: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def mi_values_(self) -> dict[str, float]:
        """Computed MI score for each evaluated feature."""
        return self._mi_values

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> MutualInformationSelector:
        """Compute MI for each feature and record which columns to drop.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series
            Target series (binary or multi-class).

        Returns
        -------
        MutualInformationSelector
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If ``y`` is ``None``.
        """
        if y is None:
            raise ValueError("y must be provided for MutualInformationSelector.fit()")

        cols = self.subset if self.subset is not None else X.columns

        self._mi_values = {col: self._compute_mi(X[col], y) for col in cols}

        below = {col for col, mi in self._mi_values.items() if mi < self.threshold}
        self._columns_to_drop = [col for col in X.columns if col in below]
        self._selected_features = [col for col in X.columns if col not in below]
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_mi(self, x: pl.Series, y: pl.Series) -> float:
        """Compute mutual information between *x* and *y*.

        Numeric *x* is first discretized into equal-width bins.
        ``None``/``NaN`` rows are dropped before computation.

        Parameters
        ----------
        x : pl.Series
            Feature series.
        y : pl.Series
            Target series.

        Returns
        -------
        float
            Mutual information value (>= 0).
        """
        x_binned = (
            self._discretize(x)
            if x.dtype in _NUMERIC_DTYPES
            else x.cast(pl.String).fill_null("__null__")
        )

        y_str = y.cast(pl.String)
        df = pl.DataFrame({"x": x_binned, "y": y_str}).drop_nulls()
        n = len(df)
        if n == 0:
            return 0.0

        joint = df.group_by(["x", "y"]).agg(pl.len().alias("n_xy"))
        px = df.group_by("x").agg(pl.len().alias("n_x"))
        py = df.group_by("y").agg(pl.len().alias("n_y"))

        joint = joint.join(px, on="x").join(py, on="y")

        mi = 0.0
        for row in joint.iter_rows(named=True):
            p_xy = row["n_xy"] / n
            p_x = row["n_x"] / n
            p_y = row["n_y"] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log(p_xy / (p_x * p_y))

        return max(0.0, mi)

    def _discretize(self, x: pl.Series) -> pl.Series:
        """Discretize a numeric series into ``n_bins`` equal-width bins.

        Parameters
        ----------
        x : pl.Series
            Numeric series.

        Returns
        -------
        pl.Series
            Integer series of bin indices (nulls preserved).
        """
        x_min = x.drop_nulls().min()
        x_max = x.drop_nulls().max()
        if x_min is None or x_max is None or x_min == x_max:
            return pl.Series([0] * len(x), dtype=pl.Int32)
        # Series.min()/.max() are typed as the broad polars `PythonLiteral | None`;
        # this method is only ever called on numeric columns (see caller), so narrow explicitly.
        x_min = float(x_min)  # type: ignore[arg-type]
        x_max = float(x_max)  # type: ignore[arg-type]
        bin_width = (x_max - x_min) / self.n_bins
        return ((x - x_min) / bin_width).cast(pl.Int32).clip(0, self.n_bins - 1)
