import math

import polars as pl
from pydantic import field_validator

from ._base_selector import _BaseSelector


class CorrelationSelector(_BaseSelector):
    """Drop redundant features by removing highly correlated ones, keeping the most important.

    For every pair of numeric features (restricted to those present in
    ``importance``) whose Pearson correlation exceeds ``max_corr``, the
    feature with the lower importance score is discarded.  The greedy pass
    iterates over pairs in column order: once a feature is marked for
    removal it is skipped for all subsequent comparisons.

    Only numeric columns (non-String, non-Boolean, non-Categorical, non-Enum)
    that appear in ``importance`` are candidates for removal.  All other
    columns are kept untouched.

    Parameters
    ----------
    importance : dict[str, float]
        Mapping of feature name to importance score.  Only columns listed
        here are considered for correlation filtering; all other columns are
        passed through unchanged.
    max_corr : float, default=0.95
        Correlation threshold above which a pair is considered redundant.
        Must be in the range ``(0, 1]``.
    use_abs : bool, default=True
        When ``True`` the absolute value of the correlation is compared
        against ``max_corr``, so strong negative correlations (e.g. −0.97)
        are treated the same as strong positive ones.  When ``False`` only
        correlations that are strictly greater than ``max_corr`` (positive)
        trigger removal.

    Attributes
    ----------
    selected_features_ : list[str]
        Column names that survive the filter (set after ``fit``).
    columns_to_drop_ : list[str]
        Column names removed because a more important correlated feature
        exists (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_selection import CorrelationSelector

    >>> X = pl.DataFrame({
    ...     "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "b": [1.1, 2.1, 3.1, 4.1, 5.1],   # nearly identical to "a"
    ...     "c": [5.0, 3.0, 1.0, 4.0, 2.0],   # independent
    ... })
    >>> importance = {"a": 0.9, "b": 0.4, "c": 0.7}

    **Example 1: Default threshold (0.95)**

    >>> selector = CorrelationSelector(importance=importance)
    >>> selector.fit(X)
    CorrelationSelector(importance={'a': 0.9, 'b': 0.4, 'c': 0.7}, max_corr=0.95, use_abs=True)
    >>> selector.columns_to_drop_
    ['b']
    >>> selector.transform(X).columns
    ['a', 'c']

    **Example 2: Stricter threshold keeps both correlated features**

    >>> selector2 = CorrelationSelector(importance=importance, max_corr=0.999)
    >>> selector2.fit(X)
    CorrelationSelector(importance={'a': 0.9, 'b': 0.4, 'c': 0.7}, max_corr=0.999, use_abs=True)
    >>> selector2.columns_to_drop_
    []

    **Example 3: use_abs=False ignores negative correlations**

    >>> X_neg = pl.DataFrame({
    ...     "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "b": [-1.0, -2.0, -3.0, -4.0, -5.0],  # perfectly negatively correlated
    ...     "c": [5.0, 3.0, 1.0, 4.0, 2.0],
    ... })
    >>> importance_neg = {"a": 0.9, "b": 0.4, "c": 0.7}
    >>> selector3 = CorrelationSelector(importance=importance_neg, max_corr=0.95, use_abs=False)
    >>> selector3.fit(X_neg)
    CorrelationSelector(importance={'a': 0.9, 'b': 0.4, 'c': 0.7}, max_corr=0.95, use_abs=False)
    >>> selector3.columns_to_drop_   # negative corr not filtered when use_abs=False
    []
    """

    importance: dict[str, float]
    max_corr: float = 0.95
    use_abs: bool = True

    @field_validator("max_corr")
    @classmethod
    def check_max_corr(cls, max_corr: float) -> float:
        if not (0 < max_corr <= 1):
            raise ValueError(f"max_corr must be in (0, 1], got {max_corr}")
        return max_corr

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "CorrelationSelector":
        """Compute pairwise Pearson correlations and record which columns to drop.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Not used; present for sklearn API compatibility.

        Returns
        -------
        CorrelationSelector
            The fitted transformer instance.
        """
        _non_numeric = {pl.String, pl.Boolean, pl.Categorical, pl.Enum}
        candidates = [
            col
            for col, dtype in zip(X.columns, X.dtypes, strict=True)
            if type(dtype) not in _non_numeric and col in self.importance
        ]

        columns_to_drop: set[str] = set()

        if len(candidates) > 1:
            pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
            corr_exprs = [
                pl.corr(candidates[i], candidates[j]).alias(f"_{i}_{j}") for i, j in pairs
            ]
            sub = X.select(candidates).cast(pl.Float64)
            corr_map: dict[tuple[int, int], float | None] = dict(
                zip(pairs, sub.select(corr_exprs).row(0), strict=True)
            )

            for i, col_i in enumerate(candidates):
                if col_i in columns_to_drop:
                    continue
                for j in range(i + 1, len(candidates)):
                    col_j = candidates[j]
                    if col_j in columns_to_drop:
                        continue
                    corr = corr_map[(i, j)]
                    if corr is None or math.isnan(corr):
                        continue
                    corr_value = abs(corr) if self.use_abs else corr
                    if corr_value > self.max_corr:
                        if self.importance[col_i] >= self.importance[col_j]:
                            columns_to_drop.add(col_j)
                        else:
                            columns_to_drop.add(col_i)
                            break  # col_i is removed; skip remaining pairs with it

        self._columns_to_drop = [col for col in X.columns if col in columns_to_drop]
        self._selected_features = [col for col in X.columns if col not in columns_to_drop]
        return self
