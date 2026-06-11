from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


def _compute_psi(reference: pl.Series, current: pl.Series, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric series.

    Bins are derived from the reference distribution.  A small epsilon is
    added to proportions before taking the log to avoid division by zero.

    Parameters
    ----------
    reference : pl.Series
        Reference (training-time) distribution.
    current : pl.Series
        Current (inference-time) distribution.
    n_bins : int, default=10
        Number of quantile-based bins to use.

    Returns
    -------
    float
        PSI value.  Conventionally: <0.1 stable, 0.1–0.25 moderate shift,
        >0.25 significant shift.
    """
    eps = 1e-8

    ref_arr = reference.drop_nulls().to_numpy()
    cur_arr = current.drop_nulls().to_numpy()

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return 0.0

    # Derive bin edges from quantiles of the reference distribution
    quantiles = [i / n_bins for i in range(n_bins + 1)]
    breaks = sorted(set(float(reference.quantile(q)) for q in quantiles))

    if len(breaks) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_arr, bins=breaks)
    cur_counts, _ = np.histogram(cur_arr, bins=breaks)

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class PSIFilter(_BaseTransformer):
    """Drop columns whose Population Stability Index exceeds a threshold.

    PSI quantifies how much a feature's distribution has shifted between a
    reference dataset (typically training data) and the current dataset.
    High PSI signals distributional drift; such features are unreliable at
    inference time and are dropped.

    PSI interpretation:

    - PSI < 0.10 — stable, no significant change
    - 0.10 ≤ PSI < 0.25 — moderate shift, investigate
    - PSI ≥ 0.25 — significant shift, feature is unstable

    Only numeric (Float64, Float32, Int64, Int32) columns are evaluated for
    PSI.  Non-numeric columns are always kept.

    Parameters
    ----------
    reference_df : pl.DataFrame
        Reference DataFrame whose distributions define the baseline.
    threshold : float, default=0.2
        Maximum PSI allowed.  Columns with PSI strictly above this value
        are dropped.
    n_bins : int, default=10
        Number of quantile-based bins used when computing PSI.
    subset : list[str] or None, default=None
        Numeric columns to evaluate.  If None, all numeric columns shared
        between ``reference_df`` and the DataFrame passed to ``fit`` are used.

    Attributes
    ----------
    psi_scores_ : dict[str, float]
        PSI score for each evaluated column (set after ``fit``).
    columns_to_drop_ : list[str]
        Columns dropped because their PSI exceeded the threshold.
    selected_features_ : list[str]
        Columns kept after filtering.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_selection import PSIFilter

    >>> reference = pl.DataFrame({
    ...     "stable": [float(i % 10) for i in range(100)],
    ...     "drifted": [float(i) for i in range(100)],
    ... })
    >>> current = pl.DataFrame({
    ...     "stable":  [float(i % 10) for i in range(100)],
    ...     "drifted": [float(i + 200) for i in range(100)],
    ... })
    >>> selector = PSIFilter(reference_df=reference, threshold=0.2)
    >>> selector.fit(current)
    >>> X_transformed = selector.transform(current)
    """

    reference_df: pl.DataFrame
    threshold: float = 0.2
    n_bins: Annotated[int, Field(ge=2)] = 10
    subset: list[str] | None = None

    _psi_scores: dict[str, float] = PrivateAttr(default_factory=dict)
    _columns_to_drop: list[str] = PrivateAttr(default_factory=list)
    _selected_features: list[str] = PrivateAttr(default_factory=list)

    @property
    def psi_scores_(self) -> dict[str, float]:
        return self._psi_scores

    @property
    def columns_to_drop_(self) -> list[str]:
        return self._columns_to_drop

    @property
    def selected_features_(self) -> list[str]:
        return self._selected_features

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "PSIFilter":
        """Compute PSI for each numeric column against the reference DataFrame.

        Parameters
        ----------
        X : pl.DataFrame
            Current DataFrame to compare against ``reference_df``.
        y : pl.Series, default=None
            Not used; present for sklearn compatibility.

        Returns
        -------
        PSIFilter
            The fitted transformer instance.
        """
        numeric_dtypes = {pl.Float64, pl.Float32, pl.Int64, pl.Int32}

        if self.subset is None:
            ref_numeric = {
                col for col, dtype in self.reference_df.schema.items() if dtype in numeric_dtypes
            }
            cur_numeric = {col for col, dtype in X.schema.items() if dtype in numeric_dtypes}
            cols_to_evaluate = sorted(ref_numeric & cur_numeric)
        else:
            cols_to_evaluate = self.subset

        self._psi_scores = {
            col: _compute_psi(self.reference_df[col], X[col], n_bins=self.n_bins)
            for col in cols_to_evaluate
        }

        self._columns_to_drop = [
            col for col, psi in self._psi_scores.items() if psi > self.threshold
        ]
        self._selected_features = [col for col in X.columns if col not in self._columns_to_drop]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Drop high-PSI columns from the DataFrame.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with high-PSI columns removed.
        """
        if not self._columns_to_drop:
            return X
        return X.drop(self._columns_to_drop)
