# SPDX-License-Identifier: Apache-2.0
"""Extended benchmark matrix: rows x features x missingness x cardinality x threads.

Usage
-----
    python benchmarks/run_benchmarks.py           # baseline 50k/500k, if not already run
    python benchmarks/run_matrix_benchmarks.py

Follows up on the future-work item in paper/gators_paper.md (Section 9, item 3):
a controlled, thread-aware, wider-range benchmark, run and reported as actual
measurements rather than a proposed design. Writes one CSV per axis plus a
combined markdown summary to ``benchmarks/results/matrix/``.

Design notes (read before adding an axis)
------------------------------------------
A full factorial grid across every axis (rows x features x missingness x
cardinality x threads x libraries) is combinatorially intractable on a single
laptop-class machine, and some cells simply do not fit in memory on typical
development hardware -- e.g. 10,000,000 rows x 1,000 float64 columns is 80GB
for one array, before a second library's copy or a one-hot expansion. We
instead run a one-factor-at-a-time (OFAT) design: each axis is swept while
holding the others at a fixed baseline. This is a deliberate, documented scope
decision, not an oversight: OFAT cannot detect interaction effects between
axes (e.g. "does high cardinality hurt more at low thread count"), and the row
range (10^4-10^6) and feature range (~6-80 columns) are both narrower than the
original 10^4-10^7 / 10-1,000 proposal, for the same memory-ceiling reason.
Both limitations are stated again in benchmarks/README.md and in the paper.

This script assumes the per-case correctness already established by
``check_parity.py`` continues to hold as the input shape changes (it measures
speed, not correctness, across the swept configurations).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets import make_frames
from benchmarks.registry import build_cases
from benchmarks.run_benchmarks import RESULTS_DIR, run_case, time_fit_transform

warnings.filterwarnings("ignore")

# Baseline configuration held fixed while sweeping one axis at a time.
BASELINE_N_NUMERIC = 8
BASELINE_N_CATEGORICAL = 4
BASELINE_CARDINALITY = 20
BASELINE_NULL_FRAC = 0.10

ROW_SWEEP_NEW_SIZES = [10_000, 1_000_000]
ROW_SWEEP_EXISTING_SIZES = [50_000, 500_000]  # reused from benchmarks/run_benchmarks.py output

FEATURE_SWEEP_ROWS = 200_000
FEATURE_TIERS = [(4, 2), (8, 4), (40, 20), (60, 20)]  # (n_numeric, n_categorical)

MISSINGNESS_SWEEP_ROWS = 200_000
MISSINGNESS_VALUES = [0.0, 0.01, 0.10, 0.50]

CARDINALITY_SWEEP_ROWS = 50_000  # kept small: cardinality=1000 + OneHotEncoder is memory-heavy
CARDINALITY_VALUES = [5, 20, 100, 1000]

THREAD_SWEEP_ROWS = [50_000, 500_000]  # matches ROW_SWEEP_EXISTING_SIZES exactly, so
# sklearn/feature-engine's (already single-threaded) numbers can be reused as-is.
THREAD_SWEEP_CASES = ["NumericImputer (mean)", "OneHotEncoder", "TargetEncoder"]
THREAD_CONFIGS: list[int | None] = [1, None]  # None = default (all cores)


def run_all_cases(cases, X_polars, X_pandas, y_polars, y_pandas, numeric_cols, categorical_cols) -> list[dict]:
    """Mirrors the per-case column selection in run_benchmarks.main()'s loop body."""
    X_polars_clean = X_polars.with_columns(pl.col(numeric_cols).fill_null(0.0)) if numeric_cols else X_polars
    X_pandas_clean = X_pandas.copy()
    if numeric_cols:
        X_pandas_clean[numeric_cols] = X_pandas_clean[numeric_cols].fillna(0.0)

    rows = []
    for case in cases:
        if case.kind == "numeric" and case.name != "NumericImputer (mean)":
            pl_X, pd_X = X_polars_clean.select(numeric_cols), X_pandas_clean[numeric_cols]
        elif case.kind == "numeric":
            pl_X, pd_X = X_polars.select(numeric_cols), X_pandas[numeric_cols]
        else:
            pl_X, pd_X = X_polars.select(categorical_cols), X_pandas[categorical_cols]
        rows.append(run_case(case, pl_X, pd_X, y_polars, y_pandas))
    return rows


def sweep_rows() -> list[dict]:
    print("=== Row-count sweep ===", file=sys.stderr)
    out = []
    for n_rows in ROW_SWEEP_NEW_SIZES:
        X_pandas, X_polars, y_pandas, y_polars = make_frames(
            n_rows,
            n_numeric=BASELINE_N_NUMERIC,
            n_categorical=BASELINE_N_CATEGORICAL,
            cardinality=BASELINE_CARDINALITY,
            null_frac=BASELINE_NULL_FRAC,
            seed=0,
        )
        numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
        categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]
        cases = build_cases(numeric_cols, categorical_cols)
        for row in run_all_cases(cases, X_polars, X_pandas, y_polars, y_pandas, numeric_cols, categorical_cols):
            row["n_rows"] = n_rows
            out.append(row)
            print(f"  [{n_rows:,} rows] {row['transformer']}: gators={row['gators_s']:.4f}s", file=sys.stderr)

    for n_rows in ROW_SWEEP_EXISTING_SIZES:
        path = RESULTS_DIR / f"results_{n_rows}.csv"
        if not path.exists():
            print(f"  ({path} not found -- run run_benchmarks.py first; skipping {n_rows:,} rows)", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df["n_rows"] = n_rows
        out.extend(df.to_dict("records"))
    return out


def sweep_features() -> list[dict]:
    print("=== Feature-count sweep ===", file=sys.stderr)
    out = []
    for n_numeric, n_categorical in FEATURE_TIERS:
        X_pandas, X_polars, y_pandas, y_polars = make_frames(
            FEATURE_SWEEP_ROWS,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            cardinality=BASELINE_CARDINALITY,
            null_frac=BASELINE_NULL_FRAC,
            seed=0,
        )
        numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
        categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]
        cases = build_cases(numeric_cols, categorical_cols)
        for row in run_all_cases(cases, X_polars, X_pandas, y_polars, y_pandas, numeric_cols, categorical_cols):
            row["n_numeric"] = n_numeric
            row["n_categorical"] = n_categorical
            row["n_features"] = n_numeric + n_categorical
            out.append(row)
            print(
                f"  [{n_numeric}+{n_categorical} cols] {row['transformer']}: gators={row['gators_s']:.4f}s",
                file=sys.stderr,
            )
    return out


def sweep_missingness() -> list[dict]:
    print("=== Missingness sweep (NumericImputer only -- other cases run on pre-cleaned data) ===", file=sys.stderr)
    out = []
    for null_frac in MISSINGNESS_VALUES:
        X_pandas, X_polars, y_pandas, y_polars = make_frames(
            MISSINGNESS_SWEEP_ROWS,
            n_numeric=BASELINE_N_NUMERIC,
            n_categorical=BASELINE_N_CATEGORICAL,
            cardinality=BASELINE_CARDINALITY,
            null_frac=null_frac,
            seed=0,
        )
        numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
        categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]
        cases = [c for c in build_cases(numeric_cols, categorical_cols) if c.name == "NumericImputer (mean)"]
        for row in run_all_cases(cases, X_polars, X_pandas, y_polars, y_pandas, numeric_cols, categorical_cols):
            row["null_frac"] = null_frac
            out.append(row)
            print(f"  [null_frac={null_frac}] gators={row['gators_s']:.4f}s", file=sys.stderr)
    return out


def sweep_cardinality() -> list[dict]:
    print("=== Cardinality sweep (categorical encoders only, single categorical column) ===", file=sys.stderr)
    out = []
    for cardinality in CARDINALITY_VALUES:
        X_pandas, X_polars, y_pandas, y_polars = make_frames(
            CARDINALITY_SWEEP_ROWS,
            n_numeric=BASELINE_N_NUMERIC,
            n_categorical=1,
            cardinality=cardinality,
            null_frac=BASELINE_NULL_FRAC,
            seed=0,
        )
        numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
        categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]
        cases = [c for c in build_cases(numeric_cols, categorical_cols) if c.kind == "categorical"]
        for row in run_all_cases(cases, X_polars, X_pandas, y_polars, y_pandas, numeric_cols, categorical_cols):
            row["cardinality"] = cardinality
            out.append(row)
            print(f"  [cardinality={cardinality}] {row['transformer']}: gators={row['gators_s']:.4f}s", file=sys.stderr)
    return out


def _thread_worker(case_name: str, n_rows: int) -> None:
    """Subprocess entry point: times one case in a fresh process so ``POLARS_MAX_THREADS``
    (set by the parent via env, before this process's first Polars import) is actually honored --
    Polars' global thread pool is initialized once per process and ignores later changes."""
    X_pandas, X_polars, _, y_polars = make_frames(
        n_rows,
        n_numeric=BASELINE_N_NUMERIC,
        n_categorical=BASELINE_N_CATEGORICAL,
        cardinality=BASELINE_CARDINALITY,
        null_frac=BASELINE_NULL_FRAC,
        seed=0,
    )
    numeric_cols = [c for c in X_pandas.columns if c.startswith("num_")]
    categorical_cols = [c for c in X_pandas.columns if c.startswith("cat_")]
    cases = {c.name: c for c in build_cases(numeric_cols, categorical_cols)}
    case = cases[case_name]

    if case.kind == "numeric" and case.name != "NumericImputer (mean)":
        X = X_polars.with_columns(pl.col(numeric_cols).fill_null(0.0)).select(numeric_cols)
    elif case.kind == "numeric":
        X = X_polars.select(numeric_cols)
    else:
        X = X_polars.select(categorical_cols)
    y = y_polars if case.needs_y else None

    gators_s = time_fit_transform(case.gators_factory, X, y)
    print(json.dumps({"gators_s": gators_s}))


def _time_gators_subprocess(case_name: str, n_rows: int, max_threads: int | None) -> float:
    env = os.environ.copy()
    if max_threads is None:
        env.pop("POLARS_MAX_THREADS", None)
    else:
        env["POLARS_MAX_THREADS"] = str(max_threads)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--thread-worker", case_name, str(n_rows)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])["gators_s"]


def sweep_threads() -> list[dict]:
    print(
        "=== Thread-count sweep (gators/Polars only; sklearn/feature-engine already "
        "single-threaded by default, reused from the existing baseline) ===",
        file=sys.stderr,
    )
    existing: dict[int, pd.DataFrame] = {}
    for n_rows in THREAD_SWEEP_ROWS:
        path = RESULTS_DIR / f"results_{n_rows}.csv"
        if path.exists():
            existing[n_rows] = pd.read_csv(path).set_index("transformer")
        else:
            print(f"  ({path} not found -- run run_benchmarks.py first for the sklearn/feature-engine reference)",
                  file=sys.stderr)

    out = []
    for n_rows in THREAD_SWEEP_ROWS:
        for case_name in THREAD_SWEEP_CASES:
            for max_threads in THREAD_CONFIGS:
                label = max_threads or "default"
                print(f"  [{n_rows:,} rows, threads={label}] {case_name}...", file=sys.stderr)
                gators_s = _time_gators_subprocess(case_name, n_rows, max_threads)
                row = {"n_rows": n_rows, "transformer": case_name, "threads": label, "gators_s": gators_s}
                if n_rows in existing and case_name in existing[n_rows].index:
                    ref = existing[n_rows].loc[case_name]
                    row["sklearn_s"] = ref.get("sklearn_s")
                    row["feature_engine_s"] = ref.get("feature_engine_s")
                out.append(row)
    return out


def df_to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free markdown table renderer (avoids requiring the `tabulate` package)."""
    header = "| " + " | ".join(df.columns) + " |\n"
    header += "|" + "|".join("---:" for _ in df.columns) + "|\n"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float) and not pd.isna(v):
                cells.append(f"{v:.4f}")
            else:
                cells.append("n/a" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(lines) + "\n"


def main() -> None:
    matrix_dir = RESULTS_DIR / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    axes = {
        "rows": sweep_rows(),
        "features": sweep_features(),
        "missingness": sweep_missingness(),
        "cardinality": sweep_cardinality(),
        "threads": sweep_threads(),
    }

    summary_parts = [
        "# Extended benchmark matrix\n",
        f"Generated on {platform.platform()}, Python {platform.python_version()}.\n",
        "\nOne-factor-at-a-time sweeps around a fixed baseline "
        f"({BASELINE_N_NUMERIC} numeric + {BASELINE_N_CATEGORICAL} categorical columns, "
        f"cardinality={BASELINE_CARDINALITY}, null_frac={BASELINE_NULL_FRAC}); see "
        "benchmarks/README.md for why this is OFAT rather than a full factorial grid, "
        "and for the row/feature range actually covered versus the original proposal.\n",
    ]

    for axis_name, rows in axes.items():
        df = pd.DataFrame(rows)
        df.to_csv(matrix_dir / f"{axis_name}.csv", index=False)
        summary_parts.append(f"\n## {axis_name.capitalize()} sweep\n\n")
        summary_parts.append(df_to_markdown(df))

    summary = "".join(summary_parts)
    (matrix_dir / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--thread-worker":
        _thread_worker(sys.argv[2], int(sys.argv[3]))
        raise SystemExit(0)
    main()
