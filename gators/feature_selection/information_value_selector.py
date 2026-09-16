import polars as pl
from pydantic import PrivateAttr

from ..discretizers._base_discretizer import _BaseDiscretizer
from ._base_selector import _BaseSelector
from .information_value import compute_iv


class InformationValueSelector(_BaseSelector):
    """Drop columns whose Information Value falls below a threshold.

    Computes the IV for each categorical (String/Categorical/Enum) column
    and drops those whose IV is below ``threshold``.  When a ``discretizer``
    is provided, numeric columns are first binned by the discretizer before
    IV computation; the discretized values are used only for scoring and are
    **not** applied during ``transform`` — numeric columns that survive the
    threshold remain numeric in the output.  Without a discretizer, numeric
    columns are excluded from IV computation and are always kept.

    Parameters
    ----------
    threshold : float, default=0.02
        Minimum IV required to keep a column.  Columns with IV strictly
        below this value are dropped.
    regularization : float, default=0.01
        Regularization applied to WOE/IV calculation to avoid division by zero.
    discretizer : _BaseDiscretizer or None, default=None
        Optional discretizer used to bin numeric columns before computing IV.
        When ``None``, numeric columns are excluded from IV computation and
        always kept (backward-compatible behaviour).

    Attributes
    ----------
    selected_features_ : list[str]
        All column names that survive the threshold (set after ``fit``).
    columns_to_drop_ : list[str]
        Column names dropped because their IV was too low (set after ``fit``).
    iv_values_ : dict[str, float]
        Mapping of feature name to its computed IV value (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_selection import InformationValueSelector

    >>> X = pl.DataFrame({
    ...     "cat_strong": ["a", "b", "a", "b", "a", "b"],
    ...     "cat_weak":   ["x", "x", "x", "x", "y", "y"],
    ...     "numeric":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ... })
    >>> y = pl.Series("target", [1, 0, 1, 0, 1, 0])
    >>> selector = InformationValueSelector(threshold=0.02)
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)

    With a discretizer to score numeric columns:

    >>> from gators.discretizers import EqualSizeDiscretizer
    >>> selector = InformationValueSelector(
    ...     threshold=0.02,
    ...     discretizer=EqualSizeDiscretizer(num_bins=7),
    ... )
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)  # numeric columns remain numeric
    """

    threshold: float = 0.02
    regularization: float = 0.01
    discretizer: _BaseDiscretizer | None = None

    _iv_values: dict[str, float] = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "InformationValueSelector":
        """Compute IV for each applicable column and record which to drop.

        Categorical columns are always scored.  Numeric columns are scored
        only when a ``discretizer`` is provided; the discretizer is fitted and
        applied to ``X`` internally and its output is passed to ``compute_iv``.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series
            Binary target series.

        Returns
        -------
        InformationValueSelector
            The fitted transformer instance.
        """
        if y is None:
            raise ValueError("y must be provided for InformationValueSelector.fit()")

        X_for_iv = self.discretizer.fit_transform(X) if self.discretizer is not None else X

        iv_df = compute_iv(X_for_iv, y, regularization=self.regularization)

        self._iv_values = dict(zip(iv_df["feature"].to_list(), iv_df["iv"].to_list(), strict=False))

        below_threshold = set(iv_df.filter(pl.col("iv") < self.threshold)["feature"].to_list())

        self._columns_to_drop = [col for col in X.columns if col in below_threshold]
        self._selected_features = [col for col in X.columns if col not in below_threshold]
        return self
