# SPDX-License-Identifier: Apache-2.0
"""ONNX serving benchmark: gators native (Polars) vs ONNX Runtime.

Usage
-----
    python benchmarks/run_onnx_benchmarks.py

For a handful of representative fitted pipelines (restricted to transformer
families that actually support ONNX export -- see gators/onnx_converters),
measures the latency, throughput, and peak-RSS delta of native Polars
``transform()`` versus an exported ONNX Runtime session across a batch-size
sweep, plus the serialized ONNX graph size. Writes
``benchmarks/results/onnx_results.csv`` and
``benchmarks/results/onnx_summary.md``. See ``benchmarks/README.md`` for full
methodology and caveats -- in particular, this measures per-call latency of a
single ``transform`` / ``session.run`` call at each batch size (a "one request
of size N" scenario), not a sustained-load/concurrent-request throughput test.
"""
from __future__ import annotations

import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.datasets import make_frames

warnings.filterwarnings("ignore")

try:
    import onnx
    import onnxruntime
    from gators.onnx_converters import create_session, pipeline_to_onnx, run_session
except ImportError:
    print(
        "This benchmark requires the 'onnx' and 'onnxruntime' packages.\n"
        "Install with: pip install -e '.[onnx,benchmarks]'",
        file=sys.stderr,
    )
    raise SystemExit(1)

import gators
from gators.clippers import QuantileClipper
from gators.discretizers import EqualSizeDiscretizer
from gators.encoders import OneHotEncoder
from gators.imputers import NumericImputer
from gators.pipeline import Pipeline
from gators.scalers import StandardScaler

FIT_ROWS = 50_000
SERVE_ROWS = 100_000  # must be >= max(BATCH_SIZES)
BATCH_SIZES = [1, 10, 100, 1_000, 10_000, 100_000]
N_NUMERIC = 8
N_CATEGORICAL = 4
CARDINALITY = 20
REPEATS = 5
WARMUP = 2
PARITY_ATOL = 1e-4
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
_RSS_TO_MB = (1 / 1024) if platform.system() == "Linux" else (1 / (1024 * 1024))


def peak_rss_mb() -> float:
    """Return the process's current peak (high-water-mark) RSS in MB.

    ``ru_maxrss`` is a monotonically non-decreasing high-water mark, not a
    per-call snapshot: a delta of 0 between two measurements means "this call
    did not set a new process-wide peak", not "this call used no memory".
    Treat the reported deltas as a coarse, order-of-magnitude signal, not a
    precise per-call allocation count (documented in benchmarks/README.md).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_TO_MB


def build_pipelines(numeric_cols: list[str], categorical_cols: list[str]) -> dict[str, Pipeline]:
    """Representative fitted-pipeline shapes, restricted to ONNX-exportable transformers.

    Mirrors the README quick-start style (default ``inplace=True``) so each
    step overwrites the same column names the next step expects, rather than
    the ``inplace=False`` convention ``benchmarks/registry.py`` uses for
    standalone (non-chained) comparisons against sklearn/feature-engine.
    """
    return {
        "impute_scale": Pipeline(steps=[
            ("impute", NumericImputer(strategy="mean", subset=numeric_cols)),
            ("scale", StandardScaler(subset=numeric_cols)),
        ]),
        "impute_clip_discretize": Pipeline(steps=[
            ("impute", NumericImputer(strategy="mean", subset=numeric_cols)),
            ("clip", QuantileClipper(subset=numeric_cols)),
            ("discretize", EqualSizeDiscretizer(num_bins=5, subset=numeric_cols)),
        ]),
        "impute_encode_scale": Pipeline(steps=[
            ("impute", NumericImputer(strategy="mean", subset=numeric_cols)),
            ("encode", OneHotEncoder(subset=categorical_cols)),
            ("scale", StandardScaler(subset=numeric_cols)),
        ]),
    }


def time_call(fn, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    """Return the best-of-``repeats`` wall-clock time (seconds) for a zero-arg call."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def check_parity(name: str, gators_out: pl.DataFrame, onnx_out: pl.DataFrame, atol: float = PARITY_ATOL) -> None:
    """Raise if the ONNX Runtime output does not match native Polars output.

    Same principle as ``check_parity.py``: refuse to trust a timing unless
    the two engines agree first. Non-numeric (string/categorical) columns are
    skipped here -- they are covered by the unit-level ONNX converter tests
    (``gators/onnx_converters/tests``), not re-verified in this benchmark.
    """
    common_cols = [c for c in gators_out.columns if c in onnx_out.columns]
    if not common_cols:
        raise AssertionError(f"[{name}] no overlapping output columns between engines")
    for col in common_cols:
        try:
            a = gators_out[col].to_numpy(allow_copy=True).astype(np.float64)
            b = onnx_out[col].to_numpy(allow_copy=True).astype(np.float64)
        except (TypeError, ValueError):
            continue
        if not np.allclose(a, b, atol=atol, equal_nan=True):
            raise AssertionError(f"[{name}] ONNX/Polars mismatch in column '{col}'")


def run_pipeline_benchmark(name: str, pipeline: Pipeline, X_fit: pl.DataFrame, X_serve: pl.DataFrame) -> list[dict]:
    print(f"[onnx-bench] fitting + exporting {name}...", file=sys.stderr)
    pipeline.fit(X_fit)
    model = pipeline_to_onnx(pipeline)
    graph_size_bytes = len(model.SerializeToString())
    session = create_session(model)

    check_parity(name, pipeline.transform(X_serve.head(1_000)), run_session(session, X_serve.head(1_000)))

    rows = []
    for batch_size in BATCH_SIZES:
        X_batch = X_serve.head(batch_size)
        print(f"[onnx-bench]   {name} @ batch_size={batch_size:,}...", file=sys.stderr)

        rss_before = peak_rss_mb()
        gators_s = time_call(lambda: pipeline.transform(X_batch))
        gators_rss_delta = peak_rss_mb() - rss_before

        rss_before = peak_rss_mb()
        onnx_s = time_call(lambda: run_session(session, X_batch))
        onnx_rss_delta = peak_rss_mb() - rss_before

        rows.append({
            "pipeline": name,
            "batch_size": batch_size,
            "gators_s": gators_s,
            "onnx_s": onnx_s,
            "onnx_vs_gators": (gators_s / onnx_s) if onnx_s else None,
            "gators_rows_per_s": (batch_size / gators_s) if gators_s else None,
            "onnx_rows_per_s": (batch_size / onnx_s) if onnx_s else None,
            "gators_peak_rss_delta_mb": gators_rss_delta,
            "onnx_peak_rss_delta_mb": onnx_rss_delta,
            "onnx_graph_size_bytes": graph_size_bytes,
        })
    return rows


def format_ratio(value) -> str:
    return f"{value:.2f}x" if value is not None else "n/a"


def format_seconds(value) -> str:
    return f"{value:.6f}" if value is not None else "n/a"


def format_int(value) -> str:
    return f"{value:,.0f}" if value is not None else "n/a"


def to_markdown(rows: list[dict], name: str) -> str:
    graph_size = rows[0]["onnx_graph_size_bytes"]
    header = (
        f"\n### {name}\n\n"
        f"Serialized ONNX graph size: {graph_size:,} bytes.\n\n"
        "| Batch size | gators (s) | onnx (s) | onnx vs gators | gators (rows/s) | onnx (rows/s) "
        "| gators \u0394peak-RSS (MB) | onnx \u0394peak-RSS (MB) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        "| {b:,} | {g} | {o} | {r} | {gr} | {or_} | {gm:.1f} | {om:.1f} |".format(
            b=r["batch_size"],
            g=format_seconds(r["gators_s"]),
            o=format_seconds(r["onnx_s"]),
            r=format_ratio(r["onnx_vs_gators"]),
            gr=format_int(r["gators_rows_per_s"]),
            or_=format_int(r["onnx_rows_per_s"]),
            gm=r["gators_peak_rss_delta_mb"],
            om=r["onnx_peak_rss_delta_mb"],
        )
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_pandas_fit, X_polars_fit, _, _ = make_frames(
        FIT_ROWS, n_numeric=N_NUMERIC, n_categorical=N_CATEGORICAL, cardinality=CARDINALITY, seed=0,
    )
    X_pandas_serve, X_polars_serve, _, _ = make_frames(
        SERVE_ROWS, n_numeric=N_NUMERIC, n_categorical=N_CATEGORICAL, cardinality=CARDINALITY, seed=1,
    )
    numeric_cols = [c for c in X_pandas_fit.columns if c.startswith("num_")]
    categorical_cols = [c for c in X_pandas_fit.columns if c.startswith("cat_")]

    pipelines = build_pipelines(numeric_cols, categorical_cols)

    all_rows: list[dict] = []
    summary_parts = [
        "# ONNX serving benchmark results\n",
        f"Generated on {platform.platform()}, Python {platform.python_version()}.\n\n",
        f"`gators` {gators.__version__}, `polars` {pl.__version__}, "
        f"`onnx` {onnx.__version__}, `onnxruntime` {onnxruntime.__version__}.\n",
        f"\nFit rows: {FIT_ROWS:,}. Serving rows sampled independently (seed=1) up to "
        f"{SERVE_ROWS:,}; each batch size is the first N rows of that serving set.\n",
        "\nAll parity checks (ONNX Runtime vs. native Polars `transform`, "
        f"atol={PARITY_ATOL:g}) passed before any timing below was recorded.\n",
    ]

    for name, pipeline in pipelines.items():
        rows = run_pipeline_benchmark(name, pipeline, X_polars_fit, X_polars_serve)
        all_rows.extend(rows)
        summary_parts.append(to_markdown(rows, name))

    import pandas as pd

    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "onnx_results.csv", index=False)
    summary = "".join(summary_parts)
    (RESULTS_DIR / "onnx_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
