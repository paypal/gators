"""Benchmark runner: gators vs scikit-learn vs feature-engine on synthetic data.

Usage
-----
    python benchmarks/run_benchmarks.py

Writes one CSV per dataset size to ``benchmarks/results/`` and prints a markdown
summary table to stdout. See ``benchmarks/README.md`` for methodology and caveats.
"""
from __future__ import annotations

import platform
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets import make_frames
from benchmarks.registry import BenchmarkCase, build_cases

warnings.filterwarnings("ignore")

ROW_SIZES = [50_000, 500_000]
N_NUMERIC = 8
N_CATEGORICAL = 4
CARDINALITY = 20
REPEATS = 3
WARMUP = 1
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def time_fit_transform(factory, X, y=None, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    """Return the best-of-``repeats`` wall-clock time (seconds) for fit + transform."""
    for _ in range(warmup):
        est = factory()
        est.fit(X, y) if y is not None else est.fit(X)
        est.transform(X)

    best = float("inf")
    for _ in range(repeats):
        est = factory()
        t0 = time.perf_counter()
        est.fit(X, y) if y is not None else est.fit(X)
        est.transform(X)
        best = min(best, time.perf_counter() - t0)
    return best


def run_case(
    case: BenchmarkCase,
    X_polars: pl.DataFrame,
    X_pandas: pd.DataFrame,
    y_polars: pl.Series,
    y_pandas: pd.Series,
) -> dict:
    y_pl = y_polars if case.needs_y else None
    y_pd = y_pandas if case.needs_y else None

    gators_time = time_fit_transform(case.gators_factory, X_polars, y_pl)

    sklearn_time = (
        time_fit_transform(case.sklearn_factory, X_pandas, y_pd)
        if case.sklearn_factory is not None
        else None
    )
    feature_engine_time = (
        time_fit_transform(case.feature_engine_factory, X_pandas, y_pd)
        if case.feature_engine_factory is not None
        else None
    )

    return {
        "transformer": case.name,
        "gators_s": gators_time,
        "sklearn_s": sklearn_time,
        "feature_engine_s": feature_engine_time,
        "speedup_vs_sklearn": (sklearn_time / gators_time) if sklearn_time else None,
        "speedup_vs_feature_engine": (
            (feature_engine_time / gators_time) if feature_engine_time else None
        ),
    }


def format_speedup(value) -> str:
    return f"{value:.1f}x" if value is not None else "n/a"


def format_seconds(value) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def to_markdown(rows: list[dict], n_rows: int) -> str:
    header = (
        f"\n### {n_rows:,} rows\n\n"
        "| Transformer | gators (s) | scikit-learn (s) | feature-engine (s) "
        "| speedup vs sklearn | speedup vs feature-engine |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        "| {t} | {g} | {sk} | {fe} | {su} | {fu} |".format(
            t=r["transformer"],
            g=format_seconds(r["gators_s"]),
            sk=format_seconds(r["sklearn_s"]),
            fe=format_seconds(r["feature_engine_s"]),
            su=format_speedup(r["speedup_vs_sklearn"]),
            fu=format_speedup(r["speedup_vs_feature_engine"]),
        )
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_parts = [
        "# Benchmark results\n",
        f"Generated on {platform.platform()}, Python {platform.python_version()}.\n",
    ]

    for n_rows in ROW_SIZES:
        X_pandas, X_polars, y_pandas, y_polars = make_frames(
            n_rows,
            n_numeric=N_NUMERIC,
            n_categorical=N_CATEGORICAL,
            cardinality=CARDINALITY,
            seed=0,
        )
        numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
        categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]

        # Non-imputer numeric cases assume already-clean data (as a real pipeline
        # would impute first), since feature-engine's discretiser rejects nulls.
        X_pandas_clean = X_pandas.copy()
        X_pandas_clean[numeric_cols] = X_pandas_clean[numeric_cols].fillna(0.0)
        X_polars_clean = X_polars.with_columns(pl.col(numeric_cols).fill_null(0.0))

        cases = build_cases(numeric_cols, categorical_cols)
        rows = []
        for case in cases:
            if case.kind == "numeric" and case.name != "NumericImputer (mean)":
                pl_X, pd_X = X_polars_clean.select(numeric_cols), X_pandas_clean[numeric_cols]
            elif case.kind == "numeric":
                pl_X, pd_X = X_polars.select(numeric_cols), X_pandas[numeric_cols]
            else:
                pl_X, pd_X = X_polars.select(categorical_cols), X_pandas[categorical_cols]

            print(f"[{n_rows:,} rows] running {case.name}...", file=sys.stderr)
            rows.append(run_case(case, pl_X, pd_X, y_polars, y_pandas))

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"results_{n_rows}.csv", index=False)
        summary_parts.append(to_markdown(rows, n_rows))

    summary = "".join(summary_parts)
    (RESULTS_DIR / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
