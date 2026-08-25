from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field, PrivateAttr

from ._base_selector import _BaseSelector


class PermutationImportanceSelector(_BaseSelector):
    """Drop columns whose permutation importance falls below a threshold.

    Fits ``estimator`` on the training data, then measures how much the
    model score degrades when each feature is randomly shuffled
    (permuted).  Features whose mean importance across ``n_repeats``
    permutations is below ``threshold`` are dropped.

    .. note::
        Sklearn estimators require NumPy arrays.  The ``fit`` method calls
        ``.to_numpy()`` internally — this is unavoidable and correct.

    Parameters
    ----------
    estimator : estimator object
        A fitted or unfitted sklearn-compatible estimator.  Must implement
        ``fit`` and ``score``.
    n_repeats : int, default=5
        Number of times to permute each feature.
    threshold : float, default=0.0
        Minimum mean importance drop required to keep a feature.  Features
        with mean permutation importance strictly below this value are
        dropped.  A value of ``0.0`` keeps all features that contribute
        at least marginally.

    Attributes
    ----------
    selected_features_ : list[str]
        Column names that survive the importance threshold.
    columns_to_drop_ : list[str]
        Column names dropped due to low permutation importance.
    importances_ : dict[str, float]
        Mean permutation importance for each input feature.

    Examples
    --------
    >>> import polars as pl
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gators.feature_selection import PermutationImportanceSelector

    >>> X = pl.DataFrame({
    ...     "informative": [i % 2 for i in range(100)],
    ...     "noise":       [0] * 100,
    ... })
    >>> y = pl.Series("target", [i % 2 for i in range(100)])
    >>> estimator = RandomForestClassifier(n_estimators=10, random_state=0)
    >>> selector = PermutationImportanceSelector(
    ...     estimator=estimator, n_repeats=5, threshold=0.0
    ... )
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)
    """

    estimator: object
    n_repeats: Annotated[int, Field(ge=1)] = 5
    threshold: float = 0.0

    _importances: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def importances_(self) -> dict[str, float]:
        return self._importances

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "PermutationImportanceSelector":
        """Fit estimator and compute permutation importances.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series
            Target series.

        Returns
        -------
        PermutationImportanceSelector
            The fitted transformer instance.
        """
        if y is None:
            raise ValueError("y must be provided for PermutationImportanceSelector.fit()")

        X_array = X.to_numpy()
        y_array = y.to_numpy()

        self.estimator.fit(X_array, y_array)  # type: ignore[attr-defined]
        baseline_score = self.estimator.score(X_array, y_array)  # type: ignore[attr-defined]

        rng = np.random.default_rng(0)
        mean_importances = np.zeros(X_array.shape[1])

        for col_idx in range(X_array.shape[1]):
            scores = np.empty(self.n_repeats)
            for repeat in range(self.n_repeats):
                X_permuted = X_array.copy()
                X_permuted[:, col_idx] = rng.permutation(X_permuted[:, col_idx])
                scores[repeat] = self.estimator.score(X_permuted, y_array)  # type: ignore[attr-defined]
            mean_importances[col_idx] = baseline_score - scores.mean()

        self._importances = {col: float(mean_importances[i]) for i, col in enumerate(X.columns)}
        self._selected_features = [
            col for col in X.columns if self._importances[col] >= self.threshold
        ]
        self._columns_to_drop = [
            col for col in X.columns if self._importances[col] < self.threshold
        ]
        return self
