---
title: 'Gators: Machine Learning Preprocessing and Feature Engineering on Polars with ONNX Export'
authors:
  - name: Charles Poli
    corresponding: true
    affiliation: 1
    orcid: 0009-0000-6564-2096
  - name: Shreekanthadatta Eligar
    affiliation: 1
    orcid: 0009-0009-1752-7822
affiliations:
  - name: PayPal, Inc., San Jose, CA, USA
    index: 1
date: 15 September 2026
bibliography: paper.bib
---

<!--
Draft manuscript for submission to SoftwareX, following the official
"SoftwareX article template Version 6 (March 2026)" section-for-section
(Title / Authors / Abstract / Keywords / Metadata / the 5 mandatory sections
/ Acknowledgements (optional) / References). Declaration of competing
interest, CRediT, Data availability, and generative-AI-use disclosure are
NOT part of this template body -- they are collected separately during
Elsevier's Editorial Manager submission steps; don't add them back here.
Before submission: paste this content into the actual .docx template,
delete this comment and all template instructions (italic text), and embed
figures/tables directly in the .docx as required.
-->

## Title

Gators: Machine Learning Preprocessing and Feature Engineering on Polars
with ONNX Export

## Authors

- Charles Poli (corresponding author) -- PayPal, Inc., San Jose, CA, USA -- email: cpoli374@gmail.com
- Shreekanthadatta Eligar -- PayPal, Inc., San Jose, CA, USA  -- email: seligar@paypal.com

## Abstract

Data preprocessing and feature engineering run far more often than model
training: every serving request re-executes transformations fit once at
training time, and re-implementing them for production is a common source
of technical debt and training/serving skew. `Gators` -- named for the
alligator, which looks idle but strikes in a fraction of a second -- is an
open-source Python library providing 107 `scikit-learn`-compatible
transformers across 10 categories, built on the `Polars` DataFrame engine
for multi-core, lazily-optimized execution. A fitted `Gators` pipeline
exports as a single, checker-validated ONNX graph, so the same code trains
and serves without a
hand-maintained second implementation. The library ships 2,323 tests and a
reproducible benchmark suite against `scikit-learn` and `feature-engine`.

**Keywords:** Python; machine learning; feature engineering; data
preprocessing; Polars; ONNX

## Metadata

| Nr | Code metadata description | Metadata |
|---|---|---|
| C1 | Current code version | v1.3.1 |
| C2 | Permanent link to code/repository used for this code version | <https://github.com/paypal/gators> |
| C3 | Legal code license | Apache License 2.0 |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python, Polars, Pydantic, scikit-learn, ONNX, ONNX Runtime |
| C6 | Compilation requirements, operating environments and dependencies | Python 3.10+ (3.14 recommended); `polars`, `pydantic`, `scikit-learn`, `pyarrow`; `onnx`/`onnxruntime` optional for ONNX export |
| C7 | If available, link to developer documentation/manual | <https://paypal.github.io/gators/> |
| C8 | Support email for questions | See GitHub Issues at the repository above |

## Motivation and significance

In production machine learning systems, the same preprocessing logic is
typically implemented twice: once in `pandas`/`scikit-learn` for training and
experimentation, and a second time, by hand, in whatever language or
framework serves predictions in production. Sculley et al. [1]
identify this kind of data-pipeline glue code as one of the largest sources
of technical debt in ML systems, and Breck et al. [2] show that
*training/serving skew* — computing a feature differently at training time
than at serving time — is one of the most common causes of silent production
degradation. Re-implementing the same logic twice is a recurring maintenance
and correctness burden; tying training-time preprocessing directly to a
single serving runtime removes that duplication but trades away portability
across serving stacks.

`Gators` addresses this from the software-engineering side: a single,
`scikit-learn`-compatible codebase handles both roles. It reuses the
`fit`/`transform` estimator contract introduced by `scikit-learn`
[3] — whose own `preprocessing` module and `ColumnTransformer`
cover a subset of what `Gators` implements, on NumPy [4]/
`pandas`, single-threaded by default — so existing `scikit-learn` pipelines,
tooling, and mental models transfer directly. `feature-engine` [5]
and `category_encoders` [6] extend the same contract with
broader transformer catalogues, but on the same `pandas` execution model.
`Gators` instead executes on `Polars`, a Rust-implemented
[7], multi-threaded DataFrame engine built on the Apache
Arrow columnar format [8] with a lazy query optimizer [9], and
adds a built-in export path from a fitted pipeline to a portable ONNX
inference graph — closing the training/serving gap directly rather than
requiring a second implementation or a single fixed serving runtime.
`sklearn-onnx` [10] converts individual `scikit-learn` estimators
to ONNX one at a time; `Gators` exports an entire fitted pipeline as one
self-contained, checker-validated `ModelProto`. TensorFlow Transform
[11] takes the opposite architectural approach, compiling
preprocessing directly into a TensorFlow serving graph; `Gators` keeps
training-time execution framework-agnostic and generates the portable graph
only at export time, with a documented, audited boundary on which
transformers can currently make that trip (Section "Software description").

## Software description

### Software architecture

Every transformer subclasses a common `_BaseTransformer`, combining
Pydantic's [12] `BaseModel` (typed, validated, keyword-only
configuration) with `scikit-learn`'s `BaseEstimator`/`TransformerMixin`
(`get_params`, `set_params`, pipeline compatibility). `fit(X, y=None)`
computes and stores statistics as private attributes and returns `self`;
`transform(X)` applies those statistics and returns a new `Polars`
DataFrame. A shared `__init_subclass__` hook gives every transformer
consistent `LazyFrame` materialization, a `NotFittedError` guard, and
automatic input-column/dtype bookkeeping, so that 107 independently authored
transformers share one calling convention without reimplementing it
individually. Internally, transformers batch column-wise operations into a
single Polars expression list per `with_columns`/`select` call rather than
looping and re-assigning the DataFrame column by column, which is the direct
mechanism behind the library's measured performance advantage over
`pandas`-based tooling: it lets one `transform` call compile to one query
plan that the engine can parallelize across columns and cores, instead of
many sequential ones.

`gators.pipeline.Pipeline` chains named steps exactly as
`sklearn.pipeline.Pipeline` does, remaining a valid `scikit-learn` estimator
to downstream tooling. `gators.onnx_converters.pipeline_to_onnx` walks a
fitted pipeline and, for each step, invokes a per-transformer-family ONNX
converter; the resulting graph is opset-selected automatically and validated
with `onnx.checker` before being returned. All data-cleaning, clipping,
encoding, imputation, discretization, scaling, and numeric/datetime
feature-generation transformers currently convert. Nine string-feature
transformers do not — seven because they rely on variable-length
tokenization, fuzzy edit distance, or regex-group/list-aggregate semantics
that ONNX's string-tensor operator set cannot express, and two because they
require a string-length or string-slice operator absent from the standard
opset — and the six feature-selection transformers are not yet wired into
the exporter. `check_pipeline_onnx_compatibility` lets a user audit a
pipeline for this boundary *before* attempting a production export, and the
exporter itself supports both fail-fast and pass-through-unchanged behavior
for unsupported steps, rather than failing opaquely.

### Software functionalities

`Gators` provides 107 transformers, organized into 10 categories: data
cleaning, outlier clipping, categorical encoding — including CatBoost-style
target encoding [13] — numeric/string/datetime feature
generation, missing-value imputation, discretization, scaling, and feature
selection. The library is tested with 2,323 unit and ONNX-conversion tests
and ships a reproducible benchmark suite comparing it against `scikit-learn`
and `feature-engine` [5] across row counts, feature counts,
missingness, cardinality, and thread configurations.

### Sample code snippets analysis (optional)

A worked example chaining two transformers into a pipeline, fitting it, and
exporting it to ONNX is given in Section "Illustrative examples" below.

## Illustrative examples

The snippet below demonstrates the major functions covered in Section
"Software description": fitting a multi-step pipeline, auditing its ONNX
compatibility, and exporting it for serving.

```python
import polars as pl
from gators.pipeline import Pipeline
from gators.imputers import NumericImputer
from gators.encoders import WOEEncoder
from gators.onnx_converters import pipeline_to_onnx, check_pipeline_onnx_compatibility

X_train = pl.DataFrame({
    "amount": [10.0, None, 30.0, 40.0],
    "category": ["a", "b", "a", None],
})
y_train = pl.Series([0, 1, 0, 1])

pipe = Pipeline(steps=[
    ("num_impute", NumericImputer(strategy="mean")),
    ("woe_encode", WOEEncoder(subset=["category"])),
])
pipe.fit(X_train, y=y_train)
X_transformed = pipe.transform(X_train)

# Audit ONNX compatibility, then export the fitted pipeline for serving.
print(check_pipeline_onnx_compatibility(pipe))
onnx_model = pipeline_to_onnx(pipe, X_train)
```

## Impact

*(This is the main section reviewers weight most heavily; each point below
answers a specific question required by the template.)*

### New research questions

Publishing an explicit, audited boundary on
which transformer families can and cannot currently be represented as an
ONNX graph (`check_pipeline_onnx_compatibility`) gives researchers building
ONNX-based deployment tooling for classical (non-neural) ML pipelines a
concrete, reproducible starting point for asking which additional
string/tabular operators are worth adding to the ONNX standard operator set,
rather than treating "does preprocessing translate to ONNX" as an
all-or-nothing question.

### Improving pursuit of existing research questions

For practitioners
already building `scikit-learn`-style preprocessing pipelines, `Gators`
provides a drop-in, multi-threaded alternative to single-threaded
`pandas`-based preprocessing, directly reducing the wall-clock
cost of the iterate-train-evaluate loop on large tabular datasets, and
removes the need to maintain and test a second, hand-written implementation
of the same logic for production serving (Section "Motivation and
significance"). Table 1 reports minimum-of-3-timed-runs `fit`+`transform`
wall time at 500,000 rows on a synthetic DataFrame (8 numeric columns with
10% nulls, 4 categorical columns with cardinality 20), for every transformer
pair that implements the same algorithm across libraries (`n/a` denotes no
comparable implementation, not an untested or zero case). Full methodology,
the 50,000-row results, and reproduction instructions are in
[`benchmarks/README.md`](https://github.com/paypal/gators/blob/master/benchmarks/README.md).

**Table 1.** Wall-clock time (s) at 500,000 rows; environment: Apple M3 Max
(16 cores), macOS 26.6.2, Python 3.14.5, `gators` 1.3.0, `polars` 1.43.0,
`scikit-learn` 1.7.2, `feature-engine` 1.9.4, `pandas` 3.0.3.

| Transformer | gators (s) | scikit-learn (s) | feature-engine (s) | speedup vs sklearn | speedup vs feature-engine |
|---|---:|---:|---:|---:|---:|
| NumericImputer (mean) | **0.005** | 0.025 | 0.012 | 5.2x | 2.4x |
| StandardScaler | **0.002** | 0.009 | n/a | 3.8x | n/a |
| QuantileClipper | **0.005** | n/a | 0.056 | n/a | 12.3x |
| EqualSizeDiscretizer (5 bins) | **0.025** | 0.112 | 0.182 | 4.5x | 7.3x |
| OneHotEncoder | **0.045** | 0.283 | 0.320 | 6.2x | 7.0x |
| OrdinalEncoder | **0.029** | 0.267 | 0.127 | 9.1x | 4.3x |
| TargetEncoder | **0.029** | 0.442 | 0.163 | 15.0x | 5.5x |
| WOEEncoder | **0.028** | n/a | 0.181 | n/a | 6.5x |

A single row's absolute gap -- e.g. 7ms for `StandardScaler` -- is not the
point in isolation: it is the *ratio* that matters, because a production
pipeline chains several such steps per request, at request volumes where
per-call overhead is paid millions of times over. At the row counts typical
of transaction-scoring workloads (Section "Impact"), a 3-15x per-transformer
ratio directly translates into either a proportional reduction in batch
preprocessing time or, at fixed latency budgets, proportionally more
preprocessing headroom before a serving SLA is at risk -- not a one-off
millisecond saved once.

These are single-machine, single-session micro-benchmarks on synthetic data;
absolute numbers will shift on real datasets, though relative ordering tends
to be stable. `check_parity.py` asserts algorithm-level correctness
invariants for every case before any timing is trusted.

Exporting a fitted pipeline to ONNX [14] trades training-time flexibility
for a portable, dependency-light inference graph; Figure 1 quantifies that
trade-off for a representative `impute_scale` pipeline (`NumericImputer` +
`StandardScaler`) fit on 50,000 rows and served via ONNX Runtime
[15] (parity with native `transform` verified at `atol=1e-4`
before timing). At single-row and small-batch latency, the ONNX graph wins
(lower fixed overhead per call); past roughly 1,000 rows per batch, native
Polars execution overtakes and pulls away as batch size grows, since it can
parallelize the whole batch across cores instead of running a fixed graph
per call.

![Native Polars vs. ONNX Runtime throughput across batch sizes for the
`impute_scale` pipeline.](figures/onnx_vs_native_throughput.png)

**Figure 1.** Throughput (rows/s, log-log) for the `impute_scale` pipeline,
native `gators`/`Polars` vs. ONNX Runtime, across batch sizes 1-100,000.
Full per-pipeline results (including `impute_clip_discretize` and
`impute_encode_scale`) and peak-RSS measurements are in
[`benchmarks/results/onnx_summary.md`](https://github.com/paypal/gators/blob/master/benchmarks/results/onnx_summary.md).

Peak-RSS overhead is negligible for both runtimes below 10,000-row
batches (under 3 MB for every pipeline tested) and does not favor either
runtime uniformly once it becomes measurable at 100,000 rows: native
`gators` adds 34.6 MB against 51.6 MB for ONNX Runtime on `impute_scale`,
12.3 MB against 106.1 MB on `impute_clip_discretize`, and 18.0 MB against
a leaner 8.2 MB for ONNX Runtime on `impute_encode_scale`. Both stay
well within the memory budget of a single serving container at the batch
sizes typical of request-time scoring.

### Change to daily practice of users

Within PayPal's Payment Service
Provider (PSP) Data Team, adopting `Gators` replaced a workflow in which
preprocessing logic was prototyped in `pandas`/`scikit-learn` and then
manually reimplemented for production scoring; a single `Gators` pipeline
definition now serves both roles, exported to ONNX for serving.

### Extent of use

`Gators` is used in production fraud-detection and
risk-modeling pipelines at PayPal, at the transaction row counts typical of
that domain. Beyond its origin team, the public repository has accumulated
27 GitHub stars and 9 forks at the time of writing; external code
contributions remain limited to date, and there are not yet citable
third-party publications using the software — extending adoption evidence
beyond the originating team is an explicit goal of having open-sourced the
project.

### Commercial use

`Gators` is used in a commercial setting: it underpins
preprocessing for production fraud-detection and risk-modeling models at
PayPal, Inc. It has not, to date, led to the creation of any spin-off
company.

## Conclusions

`Gators` combines a `Polars`-native execution model with a
`scikit-learn`-compatible API and a principled, audited ONNX export path,
addressing training/serving skew without requiring a second, hand-maintained
implementation of the same preprocessing logic. The project is under active
development, with a documented boundary on ONNX convertibility and a public
issue tracker for reporting gaps or requesting new transformers.

## Acknowledgements

We thank Prem Thangamani for her managerial support of the development of
`Gators`, PayPal's PSP Data Team for supporting the project more broadly, and
the maintainers of `Polars`, `scikit-learn`, `Pydantic`, and `ONNX`, whose
libraries this project builds directly on.

## References

[1] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., Dennison, D. Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems (NeurIPS), vol. 28, 2015.

[2] Breck, E., Cai, S., Nielsen, E., Salib, M., Sculley, D. The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE International Conference on Big Data (Big Data), pp. 1123-1132, 2017. doi:10.1109/BigData.2017.8258038

[3] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[4] Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M.H., Brett, M., Haldane, A., del Río, J.F., Wiebe, M., Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., Oliphant, T.E. Array programming with NumPy. Nature, vol. 585, pp. 357-362, 2020. doi:10.1038/s41586-020-2649-2

[5] Galli, S. feature_engine: A Python package for feature engineering for machine learning. Journal of Open Source Software, vol. 6, no. 65, p. 3642, 2021. doi:10.21105/joss.03642

[6] McGinnis, W., Siu, C., Andre, S., Huang, H. category_encoders: A scikit-learn-contrib package of transformers for encoding categorical data. https://github.com/scikit-learn-contrib/category_encoders, 2018.

[7] Matsakis, N.D., Klock, F.S. The Rust language. ACM SIGAda Ada Letters, vol. 34, no. 3, pp. 103-104, 2014. doi:10.1145/2692956.2663188

[8] Apache Software Foundation. Apache Arrow: A cross-language development platform for in-memory data. https://arrow.apache.org, 2016.

[9] Vink, R., Polars Contributors. Polars: Fast multi-threaded, hybrid-out-of-core DataFrame library. https://www.pola.rs, 2020.

[10] ONNX Contributors. sklearn-onnx: Convert scikit-learn models to ONNX. https://github.com/onnx/sklearn-onnx, 2019.

[11] Baylor, D., Breck, E., Cheng, H.-T., Fiedel, N., Foo, C.Y., Haque, Z., Haykal, S., Ispir, M., Jain, V., Koc, L., Koo, C.Y., Lew, L., Mewald, C., Modi, A.N., Polyzotis, N., Ramesh, S., Roy, S., Whang, S.E., Wicke, M., Wilkiewicz, J., Zhang, X., Zinkevich, M. TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 1387-1395, 2017. doi:10.1145/3097983.3098021

[12] Colvin, S., Pydantic Contributors. Pydantic: Data validation using Python type hints. https://docs.pydantic.dev, 2017.

[13] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018.

[14] ONNX Contributors. ONNX: Open Neural Network Exchange. https://onnx.ai, 2019.

[15] Microsoft. ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator. https://onnxruntime.ai, 2019.
