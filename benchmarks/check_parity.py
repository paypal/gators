"""Correctness/parity checks for the benchmark suite.

Confirms gators, scikit-learn, and feature-engine are doing genuinely
equivalent, correct work -- not just fast work -- before their timings in
``run_benchmarks.py`` are trusted. Checks are algorithm-level invariants (no
nulls remain, values stay within expected bounds, bin/category counts match
the source data) rather than exact value-for-value equality, since the three
libraries use different internal category-ordering conventions.

Usage
-----
    python benchmarks/check_parity.py

Exits with status 1 if any check fails.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets import make_frames
from benchmarks.registry import build_cases

warnings.filterwarnings("ignore")

N_ROWS = 5_000
N_NUMERIC = 4
N_CATEGORICAL = 3
CARDINALITY = 10
SEED = 1


class ParityError(AssertionError):
    """Raised when an implementation fails a correctness invariant."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ParityError(message)


def check_numeric_imputer(out: np.ndarray, true_means: np.ndarray) -> None:
    _check(not np.isnan(out).any(), "NaNs remain after imputation")
    means = out.mean(axis=0)
    _check(
        np.allclose(means, true_means, rtol=1e-3, atol=1e-3),
        f"imputed column means {means} != expected {true_means}",
    )


def check_standard_scaler(out: np.ndarray) -> None:
    _check(np.allclose(out.mean(axis=0), 0.0, atol=0.05), "scaled mean is not ~0")
    _check(np.allclose(out.std(axis=0), 1.0, atol=0.1), "scaled std is not ~1")


def check_quantile_clipper(out: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    _check(
        bool((out >= lower - 1e-6).all() and (out <= upper + 1e-6).all()),
        "clipped values fall outside the [1%, 99%] quantile bounds",
    )


def check_discretizer(out: np.ndarray, num_bins: int) -> None:
    for j in range(out.shape[1]):
        n_unique = len(np.unique(out[:, j]))
        _check(
            1 < n_unique <= num_bins,
            f"column {j} has {n_unique} bins (expected <= {num_bins})",
        )


def check_onehot(out: np.ndarray, n_categorical_cols: int) -> None:
    row_sums = out.sum(axis=1)
    _check(
        np.allclose(row_sums, n_categorical_cols),
        f"one-hot row sums != {n_categorical_cols} (got {row_sums[:5]})",
    )


def check_ordinal(out: np.ndarray, n_unique_per_col: list[int]) -> None:
    _check(not pd.isna(out).any(), "NaNs in ordinal codes")
    for j, expected in enumerate(n_unique_per_col):
        n_unique = len(np.unique(out[:, j]))
        _check(n_unique == expected, f"column {j} has {n_unique} codes, expected {expected}")


def check_target_encoder(out: np.ndarray) -> None:
    out = out.astype(float)
    _check(np.isfinite(out).all(), "non-finite values in target encoding")
    _check(
        bool((out >= -1e-6).all() and (out <= 1 + 1e-6).all()),
        "target-encoded values fall outside [0, 1] for a binary target",
    )


def check_woe_encoder(out: np.ndarray) -> None:
    out = out.astype(float)
    _check(np.isfinite(out).all(), "non-finite WOE values")
    _check(out.std() > 0, "WOE encoding has zero variance (looks like a no-op)")


def run_check(case_name: str, out: np.ndarray, ctx: dict) -> None:
    if case_name == "NumericImputer (mean)":
        check_numeric_imputer(out, ctx["true_means"])
    elif case_name == "StandardScaler":
        check_standard_scaler(out)
    elif case_name == "QuantileClipper":
        check_quantile_clipper(out, ctx["lower"], ctx["upper"])
    elif case_name == "EqualSizeDiscretizer (5 bins)":
        check_discretizer(out, num_bins=5)
    elif case_name == "OneHotEncoder":
        check_onehot(out, n_categorical_cols=N_CATEGORICAL)
    elif case_name == "OrdinalEncoder":
        check_ordinal(out, ctx["n_unique_per_cat"])
    elif case_name == "TargetEncoder":
        check_target_encoder(out)
    elif case_name == "WOEEncoder":
        check_woe_encoder(out)
    else:
        raise ValueError(f"no correctness check registered for {case_name!r}")


def to_array(out) -> np.ndarray:
    return out.to_pandas().to_numpy() if isinstance(out, pl.DataFrame) else np.asarray(out)


def main() -> int:
    X_pandas, X_polars, y_pandas, y_polars = make_frames(
        N_ROWS,
        n_numeric=N_NUMERIC,
        n_categorical=N_CATEGORICAL,
        cardinality=CARDINALITY,
        seed=SEED,
    )
    numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
    categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]

    X_pandas_clean = X_pandas.copy()
    X_pandas_clean[numeric_cols] = X_pandas_clean[numeric_cols].fillna(0.0)
    X_polars_clean = X_polars.with_columns(pl.col(numeric_cols).fill_null(0.0))

    ctx = {
        "true_means": X_pandas[numeric_cols].mean().to_numpy(),
        "lower": X_pandas_clean[numeric_cols].quantile(0.01).to_numpy(),
        "upper": X_pandas_clean[numeric_cols].quantile(0.99).to_numpy(),
        "n_unique_per_cat": [X_pandas[c].nunique() for c in categorical_cols],
    }

    cases = build_cases(numeric_cols, categorical_cols)
    results: list[tuple[str, str, str, str]] = []

    for case in cases:
        if case.kind == "numeric" and case.name != "NumericImputer (mean)":
            X_pl, X_pd = X_polars_clean.select(numeric_cols), X_pandas_clean[numeric_cols]
        elif case.kind == "numeric":
            X_pl, X_pd = X_polars.select(numeric_cols), X_pandas[numeric_cols]
        else:
            X_pl, X_pd = X_polars.select(categorical_cols), X_pandas[categorical_cols]

        y_pl = y_polars if case.needs_y else None
        y_pd = y_pandas if case.needs_y else None

        for lib_name, factory, X, y in [
            ("gators", case.gators_factory, X_pl, y_pl),
            ("sklearn", case.sklearn_factory, X_pd, y_pd),
            ("feature_engine", case.feature_engine_factory, X_pd, y_pd),
        ]:
            if factory is None:
                results.append((case.name, lib_name, "SKIP", "no equivalent implementation"))
                continue
            try:
                est = factory()
                est.fit(X, y) if y is not None else est.fit(X)
                out = to_array(est.transform(X))
                run_check(case.name, out, ctx)
                results.append((case.name, lib_name, "PASS", ""))
            except ParityError as exc:
                results.append((case.name, lib_name, "FAIL", str(exc)))

    width = max(len(r[0]) for r in results)
    for name, lib, status, detail in results:
        print(f"{name:<{width}}  {lib:<15}  {status:<5}  {detail}")

    n_failed = sum(1 for r in results if r[2] == "FAIL")
    print(f"\n{len(results) - n_failed}/{len(results)} checks passed.")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
