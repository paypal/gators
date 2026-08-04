"""MICE-style iterative imputer for numeric columns."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from pydantic import PositiveInt, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer

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


class IterativeImputer(_BaseTransformer):
    """Impute missing numeric values using a MICE-style iterative algorithm.

    For each numeric column that contains missing values, a linear regression
    model is fitted using all other numeric columns as predictors.  The
    algorithm repeats for ``max_iter`` rounds; on each round the imputed
    values from the previous round are used as predictors for the next,
    gradually refining the estimates.

    Algorithm (fit)
    ---------------
    1. Compute the initial fill statistic (mean or median) for every numeric
       column.
    2. Replace all ``null`` values in the training data with those initial
       statistics.
    3. For each of ``max_iter`` rounds:
       a. For every column in ``subset`` (processed in ascending order of
          null count):

          - Select rows where the column was *originally* non-null as the
            training set.
          - Fit OLS (``numpy.linalg.lstsq``) using all other numeric columns
            as features.
          - Predict the imputed values and update those rows in the working
            copy of the data.
    4. Store the final regression coefficients.

    Algorithm (transform)
    ---------------------
    1. Initialize nulls with training statistics.
    2. For each of ``max_iter`` rounds apply the stored regression models to
       predict and update every null position.
    3. Return the DataFrame with imputed values.

    Parameters
    ----------
    max_iter : PositiveInt, default=10
        Number of full passes over the imputable columns.
    subset : list[str] or None, default=None
        Numeric columns to impute.  If ``None``, all numeric columns that
        contain at least one null in the training data are selected.
    initial_strategy : {'mean', 'median'}, default='mean'
        Statistic used to initialize missing values before the first pass.

    Attributes
    ----------
    statistics_ : dict[str, float]
        Per-column initialisation statistics (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.imputers import IterativeImputer

    >>> X = pl.DataFrame({
    ...     "age":    [25.0, 30.0, None, 40.0, 35.0],
    ...     "salary": [50.0, None,  70.0, 80.0, 60.0],
    ...     "score":  [ 5.0,  8.0,   7.0, None,  6.0],
    ... })
    >>> imputer = IterativeImputer(max_iter=5)
    >>> imputer.fit(X)
    IterativeImputer(max_iter=5, subset=['age', 'salary', 'score'], initial_strategy='mean')
    >>> imputer.transform(X)
    shape: (5, 3)
    ┌──────────┬──────────┬──────────┐
    │ age      ┆ salary   ┆ score    │
    │ ---      ┆ ---      ┆ ---      │
    │ f64      ┆ f64      ┆ f64      │
    ╞══════════╪══════════╪══════════╡
    │ 25.0     ┆ 50.0     ┆ 5.0      │
    │ 30.0     ┆ ...      ┆ 8.0      │
    │ ...      ┆ 70.0     ┆ 7.0      │
    │ 40.0     ┆ 80.0     ┆ ...      │
    │ 35.0     ┆ 60.0     ┆ 6.0      │
    └──────────┴──────────┴──────────┘
    """

    max_iter: PositiveInt = 10
    subset: list[str] | None = None
    initial_strategy: Literal["mean", "median"] = "mean"

    _statistics: dict[str, float] = PrivateAttr(default_factory=dict)
    _coefs: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _feature_cols: list[str] = PrivateAttr(default_factory=list)
    _imputation_order: list[str] = PrivateAttr(default_factory=list)
    _feat_col_idx: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def statistics_(self) -> dict[str, float]:
        """Per-column initialisation statistics (mean or median)."""
        return self._statistics

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "IterativeImputer":
        """Fit regression models for each imputable column.

        Parameters
        ----------
        X : pl.DataFrame
            Training DataFrame with numeric columns.
        y : pl.Series or None, default=None
            Ignored; present for sklearn compatibility.

        Returns
        -------
        IterativeImputer
            The fitted transformer instance.
        """
        # Feature pool: all numeric columns
        self._feature_cols = [
            col for col, dtype in zip(X.columns, X.dtypes) if dtype in _NUMERIC_DTYPES
        ]
        self._feat_col_idx = {col: i for i, col in enumerate(self._feature_cols)}

        # Determine which columns to impute
        if self.subset is None:
            impute_cols = [col for col in self._feature_cols if X[col].null_count() > 0]
        else:
            impute_cols = [col for col in self.subset if col in self._feat_col_idx]
        # Store resolved subset so transform can re-use it
        self.subset = impute_cols

        if not self._feature_cols:
            return self

        # Compute initial fill statistics
        if self.initial_strategy == "mean":
            stats = X.select([pl.col(c).mean() for c in self._feature_cols]).row(0)
        else:
            stats = X.select([pl.col(c).median() for c in self._feature_cols]).row(0)

        self._statistics = {
            col: float(stats[i] if stats[i] is not None else 0.0)
            for i, col in enumerate(self._feature_cols)
        }

        # Sort by ascending null count so simpler columns are imputed first
        null_counts = {col: X[col].null_count() for col in impute_cols}
        self._imputation_order = sorted(impute_cols, key=lambda c: null_counts[c])

        # Convert to numpy and track original null positions
        X_np = self._to_numpy(X)
        null_mask = np.isnan(X_np)  # (n_rows, n_feature_cols)

        # Initialise missing values
        for j, col in enumerate(self._feature_cols):
            X_np[null_mask[:, j], j] = self._statistics[col]

        # MICE iterations
        for _ in range(self.max_iter):
            for col in self._imputation_order:
                j = self._feat_col_idx[col]
                train_rows = ~null_mask[:, j]
                feat_idx = [k for k in range(len(self._feature_cols)) if k != j]

                if feat_idx and train_rows.any():
                    X_feat = X_np[train_rows][:, feat_idx]
                    y_col = X_np[train_rows, j]
                    X_b = np.column_stack([np.ones(len(X_feat)), X_feat])
                    coefs, _, _, _ = np.linalg.lstsq(X_b, y_col, rcond=None)
                    self._coefs[col] = coefs

                    # Update imputed values in working copy
                    pred_rows = null_mask[:, j]
                    if pred_rows.any():
                        X_pred = X_np[pred_rows][:, feat_idx]
                        X_b_pred = np.column_stack([np.ones(pred_rows.sum()), X_pred])
                        X_np[pred_rows, j] = X_b_pred @ coefs
                else:
                    # Fall back to global statistic
                    self._coefs[col] = np.array([self._statistics[col]])

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Impute missing values using the fitted regression models.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame; numeric columns may contain nulls.

        Returns
        -------
        pl.DataFrame
            DataFrame with imputed numeric columns.
        """
        if not self._feature_cols or not self.subset:
            return X

        X_np = self._to_numpy(X)
        null_mask = np.isnan(X_np)

        # Initialise with training statistics
        for j, col in enumerate(self._feature_cols):
            X_np[null_mask[:, j], j] = self._statistics[col]

        # Apply regression models iteratively
        for _ in range(self.max_iter):
            for col in self._imputation_order:
                if col not in self._coefs:
                    continue
                j = self._feat_col_idx[col]
                pred_rows = null_mask[:, j]
                if not pred_rows.any():
                    continue

                coefs = self._coefs[col]
                feat_idx = [k for k in range(len(self._feature_cols)) if k != j]

                if feat_idx and len(coefs) > 1:
                    X_pred = X_np[pred_rows][:, feat_idx]
                    X_b_pred = np.column_stack([np.ones(pred_rows.sum()), X_pred])
                    X_np[pred_rows, j] = X_b_pred @ coefs
                else:
                    X_np[pred_rows, j] = coefs[0]

        # Write imputed columns back — only columns that were in subset
        subset_set = set(self.subset if self.subset else [])
        result = X
        for j, col in enumerate(self._feature_cols):
            if col in subset_set and null_mask[:, j].any():
                result = result.with_columns(pl.Series(col, X_np[:, j], dtype=pl.Float64))

        return result

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _to_numpy(self, X: pl.DataFrame) -> np.ndarray:
        """Extract ``_feature_cols`` from *X* as a float64 numpy array.

        Null values become ``NaN``.

        Parameters
        ----------
        X : pl.DataFrame
            Source DataFrame.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows, n_feature_cols)``, dtype ``float64``.
        """
        return (
            X.select([pl.col(c).cast(pl.Float64) for c in self._feature_cols])
            .to_numpy(allow_copy=True)
            .astype(np.float64)
        )
