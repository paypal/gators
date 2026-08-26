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
