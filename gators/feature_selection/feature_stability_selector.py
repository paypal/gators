from typing import Annotated

import polars as pl
from pydantic import Field, PrivateAttr

from ._base_selector import _BaseSelector
from .feature_stability_index import feature_stability_index


class FeatureStabilitySelector(_BaseSelector):
    """Drop columns whose Feature Stability Index falls below a threshold.

    Wraps :func:`~gators.feature_selection.feature_stability_index` into a
    fit/transform interface.  The FSI measures how consistently a feature is
    selected across cross-validation folds; features with low FSI are
    considered unstable and are dropped.

    .. note::
        Sklearn estimators require NumPy arrays.  The ``fit`` method calls
        ``.to_numpy()`` internally — this is unavoidable and correct.

    Parameters
    ----------
    estimator : estimator object
        Any fitted estimator with a ``feature_importances_`` attribute
        (e.g., ``RandomForestClassifier``, ``XGBClassifier``).
    skf : sklearn splitter object
        Any sklearn cross-validation splitter (e.g., ``StratifiedKFold``).
    threshold : float, default=0.5
        Minimum FSI required to keep a column.  Columns with FSI strictly
        below this value are dropped.
    importance_threshold : float, default=0.0
        Minimum per-fold importance for a feature to count as "selected"
        in that fold.

    Attributes
    ----------
    selected_features_ : list[str]
        Column names that survive the FSI threshold.
    columns_to_drop_ : list[str]
        Column names dropped due to low FSI.
    fsi_scores_ : pl.DataFrame
        Full FSI DataFrame (feature, fsi, importance) computed during fit.

    Examples
    --------
    >>> import polars as pl
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from gators.feature_selection import FeatureStabilitySelector

    >>> X = pl.DataFrame({
    ...     "stable":   [i % 2 for i in range(100)],
    ...     "unstable": [i % 7 for i in range(100)],
    ... })
    >>> y = pl.Series("target", [i % 2 for i in range(100)])
    >>> estimator = RandomForestClassifier(n_estimators=10, random_state=0)
    >>> skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    >>> selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.5)
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)
    """

    estimator: object
    skf: object
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    importance_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    _fsi_scores: pl.DataFrame = PrivateAttr(default_factory=pl.DataFrame)

    @property
    def fsi_scores_(self) -> pl.DataFrame:
        return self._fsi_scores

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "FeatureStabilitySelector":
        """Compute FSI for each column and record which to drop.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series
            Target series for training the estimator.

        Returns
        -------
        FeatureStabilitySelector
            The fitted transformer instance.
        """
        if y is None:
            raise ValueError("y must be provided for FeatureStabilitySelector.fit()")

        self._fsi_scores = feature_stability_index(
            self.estimator,
            self.skf,
            X,
            y,
            importance_threshold=self.importance_threshold,
        )

        # feature_stability_index filters out zero-FSI features; merge back to
        # cover all input columns (missing ones implicitly have fsi=0.0).
        fsi_lookup: dict[str, float] = {
            row[0]: row[1] for row in self._fsi_scores.select(["feature", "fsi"]).iter_rows()
        }
        all_fsi = {col: fsi_lookup.get(col, 0.0) for col in X.columns}

        self._selected_features = [col for col in X.columns if all_fsi[col] >= self.threshold]
        self._columns_to_drop = [col for col in X.columns if all_fsi[col] < self.threshold]
        return self
