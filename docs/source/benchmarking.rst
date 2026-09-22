Benchmarking
============

Gators transformers are timed head-to-head against their closest `scikit-learn <https://scikit-learn.org/>`_
and `feature-engine <https://feature-engine.trainindata.com/>`_ equivalents. Only transformer pairs that
implement the same algorithm are compared, and cases with no equivalent implementation are reported as
``n/a`` rather than omitted or faked.

The full benchmark suite lives in the `benchmarks/ <https://github.com/paypal/gators/tree/main/benchmarks>`_
directory of the repository and can be reproduced on your own machine with:

.. code-block:: bash

    pip install -e ".[benchmarks]"
    python benchmarks/run_benchmarks.py

Methodology
-----------

* **Data**: synthetic DataFrame with 8 numeric columns (10% nulls) and 4 categorical columns (cardinality 20),
  generated once per row count and materialized natively for each library — a Polars ``DataFrame`` for
  Gators, a pandas ``DataFrame`` for scikit-learn / feature-engine. No conversion cost is included in either
  timing.
* **Timing**: for each case, 1 untimed warm-up run followed by 3 timed runs of ``fit(X)`` + ``transform(X)``
  on a fresh estimator instance; the **minimum** of the 3 timed runs is reported.
* **Fairness**: only algorithmically equivalent transformer pairs are compared (e.g. Gators' frequency-based
  ``OrdinalEncoder`` vs ``feature_engine.encoding.OrdinalEncoder(encoding_method="arbitrary")``, not a
  supervised variant).
* **Threading**: Gators/Polars uses all available CPU cores by default; scikit-learn and feature-engine run
  with their default (mostly single-threaded) settings. This mirrors realistic out-of-the-box usage of each
  library rather than a controlled thread-for-thread comparison.

Environment
-----------

* Hardware: Apple M3 Max, 16 cores
* OS: macOS 26.6.2 (arm64)
* Python: 3.14.5
* ``gators`` 1.3.0, ``polars`` 1.43.0, ``scikit-learn`` 1.7.2, ``feature-engine`` 1.9.4, ``pandas`` 3.0.3

Results
-------

50,000 rows
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 16 16 14 16

   * - Transformer
     - gators (s)
     - scikit-learn (s)
     - feature-engine (s)
     - speedup vs sklearn
     - speedup vs feature-engine
   * - NumericImputer (mean)
     - 0.001
     - 0.004
     - 0.002
     - 3.7x
     - 2.0x
   * - StandardScaler
     - 0.001
     - 0.001
     - n/a
     - 1.6x
     - n/a
   * - QuantileClipper
     - 0.001
     - n/a
     - 0.009
     - n/a
     - 10.1x
   * - EqualSizeDiscretizer (5 bins)
     - 0.003
     - 0.012
     - 0.025
     - 3.6x
     - 7.6x
   * - OneHotEncoder
     - 0.006
     - 0.027
     - 0.051
     - 4.5x
     - 8.4x
   * - OrdinalEncoder
     - 0.004
     - 0.026
     - 0.015
     - 5.9x
     - 3.4x
   * - TargetEncoder
     - 0.006
     - 0.045
     - 0.022
     - 7.3x
     - 3.5x
   * - WOEEncoder
     - 0.005
     - n/a
     - 0.022
     - n/a
     - 4.2x

500,000 rows
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 16 16 14 16

   * - Transformer
     - gators (s)
     - scikit-learn (s)
     - feature-engine (s)
     - speedup vs sklearn
     - speedup vs feature-engine
   * - NumericImputer (mean)
     - 0.005
     - 0.025
     - 0.012
     - 5.2x
     - 2.4x
   * - StandardScaler
     - 0.002
     - 0.009
     - n/a
     - 3.8x
     - n/a
   * - QuantileClipper
     - 0.005
     - n/a
     - 0.056
     - n/a
     - 12.3x
   * - EqualSizeDiscretizer (5 bins)
     - 0.025
     - 0.112
     - 0.182
     - 4.5x
     - 7.3x
   * - OneHotEncoder
     - 0.045
     - 0.283
     - 0.320
     - 6.2x
     - 7.0x
   * - OrdinalEncoder
     - 0.029
     - 0.267
     - 0.127
     - 9.1x
     - 4.3x
   * - TargetEncoder
     - 0.029
     - 0.442
     - 0.163
     - 15.0x
     - 5.5x
   * - WOEEncoder
     - 0.028
     - n/a
     - 0.181
     - n/a
     - 6.5x

Caveats
-------

* These are micro-benchmarks on synthetic data with fixed cardinality/null ratio; real datasets will shift
  absolute numbers (relative ordering tends to be more stable).
* Numbers are single-machine, single-run-session measurements, not averaged over multiple processes/machines
  — treat as directional, not a formal statistical benchmark.
* ``n/a`` means "no comparable implementation exists", not "0x" or "untested".

---

ONNX Serving Benchmark
=======================

A separate script measures native Polars ``transform()`` versus an exported ONNX Runtime session —
the question the fit/transform benchmark above does not answer, namely whether the ONNX export path
is actually worth using at inference time, not just numerically correct.

Methodology
-----------

* **Pipelines**: three fitted ``gators.pipeline.Pipeline`` instances restricted to ONNX-exportable
  transformers, fit on 50,000 rows and evaluated on an independently sampled 100,000-row "serving" set:

  * ``impute_scale``: ``NumericImputer`` → ``StandardScaler``
  * ``impute_clip_discretize``: ``NumericImputer`` → ``QuantileClipper`` → ``EqualSizeDiscretizer``
  * ``impute_encode_scale``: ``NumericImputer`` → ``OneHotEncoder`` → ``StandardScaler``

* **Batch-size sweep**: each pipeline is timed at batch sizes 1 / 10 / 100 / 1,000 / 10,000 / 100,000 —
  batch size 1 is the realistic "one prediction request" serving scenario; larger batches represent
  bulk scoring/offline transform.
* **Correctness first**: for every pipeline, ONNX Runtime output is checked against native Polars
  output (``atol=1e-4``) before any timing is trusted.
* **Timing**: best-of-5 wall-clock time after 2 warm-up calls, one call per batch size.
* **Memory**: ``resource.getrusage().ru_maxrss`` delta around each call — a process-wide, monotonically
  non-decreasing high-water mark, not a precise per-call allocation count. A delta of 0 means "did not
  set a new peak", not "used no memory"; treat these numbers as directional only.
* **ONNX Runtime configuration**: ``CPUExecutionProvider``, default graph optimizations
  (``ORT_ENABLE_ALL``), ``intra_op_num_threads=0`` (let ORT choose), ``inter_op_num_threads=1`` — the
  defaults ``create_session`` ships with, tuned for single-request latency rather than bulk throughput.

Reproducing
-----------

.. code-block:: bash

    pip install -e ".[onnx,benchmarks]"
    python benchmarks/run_onnx_benchmarks.py

Writes ``benchmarks/results/onnx_results.csv`` and ``benchmarks/results/onnx_summary.md``.

Environment
-----------

* Hardware/OS/Python: same machine as above (macOS 26.6.2, arm64, Python 3.14.5)
* ``gators`` 1.3.1, ``polars`` 1.43.0, ``onnx`` 1.22.0, ``onnxruntime`` 1.26.0
* Fit rows: 50,000. Serving rows sampled independently (seed=1) up to 100,000; each batch size is the
  first N rows of that serving set.

Results
-------

impute_scale
~~~~~~~~~~~~

Serialized ONNX graph size: 6,135 bytes.

.. list-table::
   :header-rows: 1
   :widths: 12 14 14 14 16 16 12 12

   * - Batch size
     - gators (s)
     - onnx (s)
     - onnx vs gators
     - gators (rows/s)
     - onnx (rows/s)
     - gators Δpeak-RSS (MB)
     - onnx Δpeak-RSS (MB)
   * - 1
     - 0.000502
     - 0.000322
     - 1.56x
     - 1,991
     - 3,102
     - 0.5
     - 0.0
   * - 10
     - 0.000573
     - 0.000317
     - 1.81x
     - 17,443
     - 31,550
     - 0.1
     - 0.0
   * - 100
     - 0.000589
     - 0.000397
     - 1.48x
     - 169,815
     - 251,783
     - 0.2
     - 0.0
   * - 1,000
     - 0.000583
     - 0.001055
     - 0.55x
     - 1,715,142
     - 947,867
     - 0.4
     - 0.0
   * - 10,000
     - 0.000565
     - 0.006680
     - 0.08x
     - 17,710,870
     - 1,496,997
     - 2.8
     - 1.4
   * - 100,000
     - 0.000856
     - 0.058543
     - 0.01x
     - 116,816,698
     - 1,708,153
     - 34.6
     - 51.6

impute_clip_discretize
~~~~~~~~~~~~~~~~~~~~~~

Serialized ONNX graph size: 19,938 bytes.

.. list-table::
   :header-rows: 1
   :widths: 12 14 14 14 16 16 12 12

   * - Batch size
     - gators (s)
     - onnx (s)
     - onnx vs gators
     - gators (rows/s)
     - onnx (rows/s)
     - gators Δpeak-RSS (MB)
     - onnx Δpeak-RSS (MB)
   * - 1
     - 0.000864
     - 0.000308
     - 2.81x
     - 1,158
     - 3,249
     - 0.0
     - 0.0
   * - 10
     - 0.000810
     - 0.000308
     - 2.63x
     - 12,349
     - 32,494
     - 0.0
     - 0.0
   * - 100
     - 0.000650
     - 0.000365
     - 1.78x
     - 153,955
     - 273,941
     - 0.0
     - 0.0
   * - 1,000
     - 0.000780
     - 0.001182
     - 0.66x
     - 1,281,366
     - 846,024
     - 0.0
     - 0.0
   * - 10,000
     - 0.001065
     - 0.009345
     - 0.11x
     - 9,393,711
     - 1,070,058
     - 0.0
     - 0.0
   * - 100,000
     - 0.003814
     - 0.090926
     - 0.04x
     - 26,221,193
     - 1,099,794
     - 12.3
     - 106.1

impute_encode_scale
~~~~~~~~~~~~~~~~~~~~

Serialized ONNX graph size: 27,709 bytes.

.. list-table::
   :header-rows: 1
   :widths: 12 14 14 14 16 16 12 12

   * - Batch size
     - gators (s)
     - onnx (s)
     - onnx vs gators
     - gators (rows/s)
     - onnx (rows/s)
     - gators Δpeak-RSS (MB)
     - onnx Δpeak-RSS (MB)
   * - 1
     - 0.002486
     - 0.000563
     - 4.41x
     - 402
     - 1,775
     - 0.0
     - 0.0
   * - 10
     - 0.002640
     - 0.000564
     - 4.68x
     - 3,788
     - 17,721
     - 0.0
     - 0.0
   * - 100
     - 0.002577
     - 0.000630
     - 4.09x
     - 38,799
     - 158,636
     - 0.0
     - 0.0
   * - 1,000
     - 0.002588
     - 0.001416
     - 1.83x
     - 386,430
     - 706,318
     - 0.0
     - 0.0
   * - 10,000
     - 0.002625
     - 0.009987
     - 0.26x
     - 3,809,644
     - 1,001,318
     - 0.0
     - 0.0
   * - 100,000
     - 0.004792
     - 0.096898
     - 0.05x
     - 20,867,386
     - 1,032,008
     - 18.0
     - 8.2

Headline result
---------------

Across all three pipelines, the same qualitative crossover appears: **ONNX Runtime is faster at small
batch sizes (single-row up to ~100–1,000 rows), native Polars is faster at large batch sizes (10,000+
rows)** — e.g. for ``impute_scale``, ONNX is ~1.6–1.8x faster than Polars at batch size 1–10, but ~12x
*slower* at batch size 10,000 and ~68x slower at 100,000.

This is the expected shape of the trade-off: Polars' advantage comes from parallelizing a query plan
across cores over a large batch, which only pays off once a batch is large enough to amortize
scheduling overhead; ONNX Runtime's advantage at small batches comes from a lighter-weight,
single-graph-execution call with none of the Python-object/query-planning overhead that ``transform()``
pays per call regardless of row count. In other words: **use the ONNX export path for low-latency,
one-row-at-a-time serving; use native Polars for bulk/batch scoring** — the two paths are
complementary, not "one strictly replaces the other".

Peak-RSS overhead is negligible for both runtimes below 10,000-row batches (under 3 MB for every
pipeline tested) and does not favor either runtime uniformly once it becomes measurable at 100,000
rows: native gators adds 34.6 MB against 51.6 MB for ONNX Runtime on ``impute_scale``, 12.3 MB against
106.1 MB on ``impute_clip_discretize``, and 18.0 MB against a leaner 8.2 MB for ONNX Runtime on
``impute_encode_scale``. Both stay well within the memory budget of a single serving container at the
batch sizes typical of request-time scoring.

ONNX caveats
------------

* Does not measure concurrent/sustained-load throughput (many simultaneous requests) — only
  single-threaded, single-request-at-a-time latency.
* Does not cover non-CPU execution providers, other thread configurations, or ``float32`` graphs (this
  benchmark uses ``pipeline_to_onnx``'s ``float64`` default).
* Pipelines containing the ``feature_generation_str`` transformers with no ONNX converter at all cannot
  take this path by construction — see the ONNX Export section of the root README.
