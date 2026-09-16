from typing import Annotated, Any

import numpy as np
import polars as pl
from pydantic import Field


def select_k_best_stable_features(
    estimator: Any, skf: Any, X: pl.DataFrame, y: pl.Series, k: Annotated[int, Field(ge=1)] = 200
) -> pl.DataFrame:
    """Select features that consistently appear in the top-k across all folds.

    For each fold, fits the estimator on the training split and selects the top-k
    features by importance. Returns the intersection — features that ranked in the
    top-k in every fold.

    Parameters
    ----------
    estimator : estimator object
        Any estimator with a ``feature_importances_`` attribute (e.g., XGBoost, RandomForest).
    skf : sklearn fold splitter object
        Any sklearn fold splitter object (e.g., StratifiedKFold, KFold) for splitting the data.
    X : pl.DataFrame
        Feature DataFrame with shape (n_samples, n_features).
    y : pl.Series
        Target series for training.
    k : Annotated[int, Field(ge=1)], default=200
        Number of top features to consider per fold. Must be at least 1.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:

        - feature: Feature name
        - importance: Average feature importance across all folds

        Contains only features that appeared in the top-k in every fold,
        sorted by importance in descending order.
    """
    X_array = X.to_numpy()
    y_array = y.to_numpy()
    features = X.columns
    importance_per_fold: dict[str, list[float]] = {col: [] for col in features}
    top_k_per_fold: list[set[str]] = []

    for train_idx, _ in skf.split(X_array, y_array):
        estimator.fit(X_array[train_idx], y_array[train_idx])
        fold_importance = dict(zip(features, estimator.feature_importances_, strict=False))
        top_k_set = set(
            pl.DataFrame({"feature": features, "importance": estimator.feature_importances_})
            .sort("importance", descending=True)[:k]["feature"]
            .to_list()
        )
        top_k_per_fold.append(top_k_set)
        for col in top_k_set:
            importance_per_fold[col].append(fold_importance[col])

    stable = set(features)
    for top_k_set in top_k_per_fold:
        stable &= top_k_set

    print(
        f"  [fsi] {len(features)} → {len(stable)} stable features (top-{k} intersection over {len(top_k_per_fold)} folds)"
    )

    avg_importance = {col: float(np.mean(importance_per_fold[col])) for col in stable}
    return pl.DataFrame(
        {
            "feature": list(avg_importance.keys()),
            "importance": list(avg_importance.values()),
        }
    ).sort("importance", descending=True)
