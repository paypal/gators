from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field, PositiveInt


def feature_stability_index(
    estimator,
    skf,
    X: pl.DataFrame,
    y: pl.Series,
    importance_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
):
    """Compute Feature Stability Index (FSI) using repeated estimator feature importance.

    Measures how consistently a feature is selected across different training folds.
    Higher FSI indicates more stable/reliable feature importance.

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
    importance_threshold : Annotated[float, Field(ge=0.0, le=1.0)], default=0.0
        Minimum importance value for a feature to be considered "selected" in a run.
        Must be between 0.0 and 1.0.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
    
        - feature: Feature name
        - fsi: Feature Stability Index (0 to 1, higher is more stable)
        - importance: Average importance across all runs
        Sorted by FSI and importance in descending order, filtered to fsi > 0.
    """
    selection_matrix = []
    importance_matrix = []

    X_array = X.to_numpy()
    y_array = y.to_numpy()

    for train_idx, _ in skf.split(X_array, y_array):
        estimator.fit(X_array[train_idx], y_array[train_idx])

        importances = estimator.feature_importances_
        importance_matrix.append(importances)

        selected = (importances > importance_threshold).astype(int)
        selection_matrix.append(selected)

    selection_matrix = np.array(selection_matrix)
    importance_matrix = np.array(importance_matrix)

    fsi_scores = selection_matrix.mean(axis=0)
    importances = importance_matrix.mean(axis=0)
    X =  pl.DataFrame(
        {"feature": X.columns, "fsi": fsi_scores, "importance": importances}
    ).sort(by=["fsi", "importance"], descending=True)
    X =  X.filter(pl.col("fsi") > 0)
    return X.sort(by=["fsi", "importance"], descending=True)
