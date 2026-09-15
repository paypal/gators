# Gators Benchmarks

Reproducible timing comparisons of `gators` against `scikit-learn` and
`feature-engine` for transformers that have a genuinely equivalent
implementation in more than one library. The goal is a credible, rerunnable
number — not a marketing multiplier.

## Methodology

- **Data**: synthetic DataFrame with 8 numeric columns (10% nulls) and 4
  categorical columns (cardinality 20), generated once per row count and
  materialized natively for each library — a polars `DataFrame` for `gators`,
  a pandas `DataFrame` for `scikit-learn` / `feature-engine`. No conversion
  cost is included in either timing.
- **Timing**: for each case, 1 untimed warm-up run followed by 3 timed runs of
  `fit(X)` + `transform(X)` on a fresh estimator instance; the **minimum** of
  the 3 timed runs is reported (reduces noise from OS scheduling/GC).
- **Fairness**: only transformer pairs that implement the same algorithm are
  compared (e.g. gators' frequency-based `OrdinalEncoder` vs
  `feature_engine.encoding.OrdinalEncoder(encoding_method="arbitrary")`, not
  against a supervised variant). Where no equivalent exists in a library
  (e.g. WOE encoding isn't in scikit-learn, `StandardScaler` isn't
  reimplemented in feature-engine), the cell is reported as `n/a` rather than
  omitted or faked.
- **Single-threaded fairness note**: gators/Polars uses all available CPU
  cores by default; scikit-learn and feature-engine here run with their
  default (mostly single-threaded) settings. This mirrors realistic
  out-of-the-box usage of each library, not a controlled thread-for-thread
  comparison.
- **Correctness first**: `check_parity.py` asserts algorithm-level
  invariants (no nulls remain, values stay in bounds, bin/category counts
  match the source data) for every case before timings are trusted. Exact
  value-for-value equality isn't checked, since each library orders
  categories/bins differently.

## Reproducing

```bash
pip install -e ".[benchmarks]"
python benchmarks/check_parity.py   # verify all implementations produce correct output first
python benchmarks/run_benchmarks.py
```

This writes `results/results_<n_rows>.csv` and `results/summary.md`, and
prints the same summary to stdout. Results depend on hardware, OS, and
library versions — always regenerate on your own machine before citing a
number.

## Environment for the numbers below

- Hardware: Apple M3 Max, 16 cores
- OS: macOS 26.6.2 (arm64)
- Python: 3.14.5
- `gators` 1.3.0, `polars` 1.43.0, `scikit-learn` 1.7.2, `feature-engine` 1.9.4, `pandas` 3.0.3

## Results

See [results/summary.md](results/summary.md) for the full generated tables
(50,000 and 500,000 row datasets). Summary at 500,000 rows:

| Transformer | gators (s) | scikit-learn (s) | feature-engine (s) | speedup vs sklearn | speedup vs feature-engine |
|---|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.005 | 0.025 | 0.012 | 5.2x | 2.4x |
| StandardScaler | 0.002 | 0.009 | n/a | 3.8x | n/a |
| QuantileClipper | 0.005 | n/a | 0.056 | n/a | 12.3x |
| EqualSizeDiscretizer (5 bins) | 0.025 | 0.112 | 0.182 | 4.5x | 7.3x |
| OneHotEncoder | 0.045 | 0.283 | 0.320 | 6.2x | 7.0x |
| OrdinalEncoder | 0.029 | 0.267 | 0.127 | 9.1x | 4.3x |
| TargetEncoder | 0.029 | 0.442 | 0.163 | 15.0x | 5.5x |
| WOEEncoder | 0.028 | n/a | 0.181 | n/a | 6.5x |

## Caveats

- These are micro-benchmarks on synthetic data with fixed cardinality/null
  ratio; real datasets will shift absolute numbers (relative ordering tends
  to be more stable).
- Numbers are single-machine, single-run-session measurements, not averaged
  over multiple processes/machines — treat as directional, not a formal
  statistical benchmark.
- `n/a` means "no comparable implementation exists", not "0x" or "untested".

## Extended benchmark matrix

A separate script sweeps rows, feature count, missingness, cardinality, and
thread count — the axes the single-point comparison above cannot speak to,
and in particular a **controlled, single-thread-vs-single-thread**
comparison that isolates how much of the headline speedup above comes from
Polars' default multi-threading rather than the transformer implementation
itself.

### Design: one-factor-at-a-time, not a full factorial grid

A full grid across every axis (rows × features × missingness × cardinality ×
threads × libraries) is combinatorially intractable on a single laptop, and
some cells don't fit in memory regardless of time budget (10,000,000 rows ×
1,000 float64 columns is 80GB for one array). We instead sweep **one axis at
a time** around a fixed baseline (8 numeric + 4 categorical columns,
cardinality 20, 10% nulls). This is a deliberate scope decision: OFAT cannot
detect interaction effects between axes, and the ranges covered here
(10⁴–10⁶ rows, ~6–80 columns) are narrower than a full 10⁴–10⁷ rows /
10–1,000 feature study would require — both stated as open follow-up work,
not hidden.

### Reproducing

```bash
python benchmarks/run_benchmarks.py           # first, if results_50000.csv / results_500000.csv don't exist yet
python benchmarks/run_matrix_benchmarks.py
```

Writes one CSV per axis plus `results/matrix/summary.md`. The row and thread
sweeps reuse `run_benchmarks.py`'s existing 50,000/500,000-row output
directly (for the sklearn/feature-engine reference numbers) rather than
re-measuring them, so results stay consistent across scripts.

### Headline result 1 — the reported speedups scale further than 2 points suggested

Extending the row axis down to 10,000 and up to 1,000,000 rows confirms and
sharpens the trend already visible at 50k/500k: every transformer's speedup
vs. scikit-learn keeps growing with row count rather than plateauing —
e.g. `TargetEncoder` goes from 4.1x (10k rows) → 7.3x (50k) → 15.0x (500k) →
18.0x (1,000,000 rows). The same pattern holds on the feature-count axis
(more columns, not just more rows, also widens the gap).

### Headline result 2 — a controlled, thread-isolated comparison meaningfully lowers the "true" speedup

This is the single most important correction in this benchmark matrix. Using
`POLARS_MAX_THREADS=1` (set before Polars' first import, via a subprocess,
since the thread pool is fixed for the process's lifetime), we re-measured
three transformers at 500,000 rows with Gators pinned to one thread, holding
scikit-learn's (already single-threaded) numbers fixed:

| Transformer | gators, 1 thread (s) | gators, default threads (s) | sklearn (s) | speedup, 1-thread-vs-1-thread | speedup, default-vs-1-thread (previously reported) |
|---|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.0053 | 0.0026 | 0.0250 | **4.72x** | 9.62x |
| OneHotEncoder | 0.0663 | 0.0362 | 0.2835 | **4.28x** | 7.83x |
| TargetEncoder | 0.0923 | 0.0230 | 0.4415 | **4.78x** | 19.20x |

Once thread count is controlled for, the three transformers converge to a
tight, consistent **~4.3–4.8x** advantage over scikit-learn — a genuinely
measured, apples-to-apples number. The previously reported default-vs-default
speedups (up to 19x for `TargetEncoder`) were real, but roughly half to
two-thirds of that number was Polars using multiple cores against a
single-threaded scikit-learn, not a 15–19x faster *implementation*. Both
numbers are legitimate and are reported together deliberately: the
default-vs-default number reflects realistic out-of-the-box usage (Section
above); the thread-isolated number is the fairer algorithmic comparison. This
directly resolves the thread-asymmetry caveat in the main methodology section
rather than leaving it as an acknowledged-but-unquantified limitation.

### Headline result 3 — `OneHotEncoder`'s advantage narrows as cardinality grows

The cardinality sweep (a single categorical column, cardinality 5 → 1,000,
50,000 rows) shows that `OneHotEncoder`'s speedup over scikit-learn is
cardinality-dependent: 3.8x faster at cardinality 5, 3.0x at cardinality 20,
2.0x at cardinality 100, and statistically indistinguishable from parity
(1.00x, 0.0115s vs. 0.0115s) at cardinality 1,000 — the only point anywhere
in this matrix where a Gators transformer does not clearly outperform
scikit-learn. `feature-engine`, by contrast, gets dramatically slower as
cardinality grows (about 120x slower than Gators at cardinality 1,000), so
the narrowing is specific to the scikit-learn comparison. We report this
trend as measured, without smoothing it into a flat "always faster" claim:
at very high cardinality, whether Gators is faster than scikit-learn is not
a given.

### What this still does not show


A full factorial grid (to detect interaction effects between axes), rows
beyond 1,000,000 or features beyond ~80 (memory-bound on this hardware),
thread sweeps for the other 5 transformer cases, multi-machine variance, and
whether `OneHotEncoder`'s narrowing advantage continues past cardinality
1,000 or appears in other wide-output transformers.
See `paper/gators_paper.md` Section 9 for the prioritized follow-up list.


## ONNX serving benchmark

A separate script measures native Polars `transform()` versus an exported
ONNX Runtime session — the question the fit/transform benchmark above does
not answer, namely whether the ONNX export path (see the root README's ONNX
Export section) is actually worth using at inference time, not just
numerically correct.

### Methodology

- **Pipelines**: three fitted `gators.pipeline.Pipeline`s restricted to
  ONNX-exportable transformers (`impute_scale`, `impute_clip_discretize`,
  `impute_encode_scale`), fit on 50,000 rows, evaluated on an independently
  sampled 100,000-row "serving" set.
- **Batch-size sweep**: each pipeline is timed at batch sizes
  1 / 10 / 100 / 1,000 / 10,000 / 100,000 — batch size 1 is the realistic
  "one prediction request" serving scenario; larger batches represent bulk
  scoring/offline transform.
- **Correctness first**: for every pipeline, ONNX Runtime output is checked
  against native Polars output (`atol=1e-4`) before any timing is trusted —
  same principle as `check_parity.py`.
- **Timing**: best-of-5 wall-clock time after 2 warm-up calls, one call per
  batch size (not chunked further).
- **Memory**: `resource.getrusage().ru_maxrss` delta around each call. This is
  a process-wide, monotonically non-decreasing high-water mark, not a precise
  per-call allocation count — a delta of 0 means "did not set a new peak",
  not "used no memory". Treat these numbers as directional only.
- **ONNX Runtime configuration**: `CPUExecutionProvider`, default graph
  optimizations (`ORT_ENABLE_ALL`), `intra_op_num_threads=0` (let ORT choose),
  `inter_op_num_threads=1` — the defaults `create_session` ships with, tuned
  for single-request latency rather than bulk throughput. A thread-swept
  comparison is listed as follow-up work, not included here.

### Reproducing

```bash
pip install -e ".[onnx,benchmarks]"
python benchmarks/run_onnx_benchmarks.py
```

Writes `results/onnx_results.csv` and `results/onnx_summary.md`, and prints
the same summary to stdout.

### Headline result

Across all three pipelines, the same qualitative crossover appears: **ONNX
Runtime is faster at small batch sizes (single-row up to ~100–1,000 rows),
native Polars is faster at large batch sizes (10,000+ rows)** — e.g. for
`impute_scale`, ONNX is ~1.6–1.8x faster than Polars at batch size 1–10, but
~12x *slower* at batch size 10,000 and ~68x slower at 100,000. See
[results/onnx_summary.md](results/onnx_summary.md) for full per-pipeline
tables.

This is the expected shape of the trade-off, not a surprise to be explained
away: Polars' advantage comes from parallelizing a query plan across cores
over a large batch, which only pays off once a batch is large enough to
amortize scheduling overhead; ONNX Runtime's advantage at small batches comes
from a lighter-weight, single-graph-execution call with none of the
Python-object/query-planning overhead that `transform()` pays per call
regardless of row count. In other words: **use the ONNX export path for
low-latency, one-row-at-a-time serving; use native Polars for bulk/batch
scoring** — the two paths are complementary, not "one strictly replaces the
other".

### What this does *not* show

- Concurrent/sustained-load throughput (many simultaneous requests) — this
  only measures single-threaded, single-request-at-a-time latency.
- Non-CPU execution providers, other thread configurations, or `float32`
  graphs (the benchmark uses `pipeline_to_onnx`'s `float64` default).
- Behavior on pipelines containing the `feature_generation_str` transformers
  that have no ONNX converter at all (Section on ONNX Export in the root
  README) — those pipelines cannot take this path by construction.
