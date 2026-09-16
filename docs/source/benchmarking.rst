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
