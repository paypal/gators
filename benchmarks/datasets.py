"""Synthetic dataset generator shared by all benchmark cases.

The same underlying data is materialized as both a pandas DataFrame (for sklearn /
feature-engine) and a polars DataFrame (for gators), so every library times its own
native, idiomatic code path rather than paying a conversion tax.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def make_frames(
    n_rows: int,
    n_numeric: int = 8,
    n_categorical: int = 4,
    cardinality: int = 50,
    null_frac: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pl.DataFrame, pd.Series, pl.Series]:
    """Generate a synthetic dataset with numeric, categorical, and target columns.

    Returns
    -------
    tuple
        ``(X_pandas, X_polars, y_pandas, y_polars)`` built from identical data.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    for i in range(n_numeric):
        col = rng.normal(loc=rng.uniform(-10, 10), scale=rng.uniform(1, 5), size=n_rows)
        if null_frac:
            col[rng.random(n_rows) < null_frac] = np.nan
        data[f"num_{i}"] = col

    categories = [f"cat_{j}" for j in range(cardinality)]
    for i in range(n_categorical):
        data[f"cat_{i}"] = rng.choice(categories, size=n_rows)

    target = rng.integers(0, 2, size=n_rows)

    X_pandas = pd.DataFrame(data)
    X_polars = pl.from_pandas(X_pandas)
    y_pandas = pd.Series(target, name="target")
    y_polars = pl.Series("target", target)
    return X_pandas, X_polars, y_pandas, y_polars
