API Reference
=============

Complete reference for all Gators transformers, organized by functionality.

.. toctree::
   :maxdepth: 2
   :hidden:

   api/data_cleaning
   api/clippers
   api/encoders
   api/feature_generation
   api/feature_generation_str
   api/feature_generation_dt
   api/imputers
   api/discretizers
   api/scalers
   api/pipeline
   api/feature_selection

Overview
--------

Gators provides **105 transformers** across 11 categories, all with a consistent sklearn-compatible API.
Each transformer implements ``.fit()`` and ``.transform()`` methods and works seamlessly with Polars DataFrames.


Data Cleaning
~~~~~~~~~~~~~

Quality filters, type casting, deduplication, rounding, and data quality transformations. **(16 transformers)**

:doc:`View Data Cleaning API <api/data_cleaning>`

Clippers
~~~~~~~~

Outlier detection and clipping strategies: Custom, Gaussian, IQR, MAD, and Quantile. **(5 transformers)**

:doc:`View Clippers API <api/clippers>`

Encoders
~~~~~~~~

Categorical encoding: Binary, CatBoost, Count, Hash, LeaveOneOut, OneHot, Ordinal, RareCategory, Target, WOE. **(10 transformers)**

:doc:`View Encoders API <api/encoders>`

Feature Generation
~~~~~~~~~~~~~~~~~~

Create numeric features: polynomial, ratios, Fourier, group aggregations, rolling windows, and custom rules. **(20 transformers)**

:doc:`View Feature Generation API <api/feature_generation>`

String Features
~~~~~~~~~~~~~~~

Extract information from text: length, patterns, n-grams, TF-IDF, regex extraction, and text statistics. **(17 transformers)**

:doc:`View String Features API <api/feature_generation_str>`

DateTime Features
~~~~~~~~~~~~~~~~~

Temporal feature engineering: cyclic encoding, holidays, business hours, time windows, and date differences. **(8 transformers)**

:doc:`View DateTime Features API <api/feature_generation_dt>`

Imputers
~~~~~~~~

Handle missing values with mean, median, mode, constant, KNN, iterative, or group-based strategies. **(6 transformers)**

:doc:`View Imputers API <api/imputers>`

Discretizers
~~~~~~~~~~~~

Bin continuous variables using equal-width, quantile, geometric, k-means, tree-based, or custom strategies. **(7 transformers)**

:doc:`View Discretizers API <api/discretizers>`

Scalers
~~~~~~~

Normalize and transform features: Standard, MinMax, Robust, BoxCox, Yeo-Johnson, Log1p, and more. **(9 transformers)**

:doc:`View Scalers API <api/scalers>`

Pipeline
~~~~~~~~

Chain multiple transformers together for streamlined preprocessing workflows.

:doc:`View Pipeline API <api/pipeline>`

Feature Selection
~~~~~~~~~~~~~~~~~

Select important features using Pearson correlation, Information Value, mutual information, permutation importance, and Population Stability Index.

:doc:`View Feature Selection API <api/feature_selection>`

ONNX Export
~~~~~~~~~~~

Export any fitted ``Pipeline`` or single transformer to a validated ONNX graph for low-latency, language-agnostic inference.

.. code-block:: python

    from gators.onnx_converters import pipeline_to_onnx, to_onnx_graph

    # Export a complete pipeline
    onnx_model = pipeline_to_onnx(fitted_pipeline)

    # Export a single transformer
    onnx_model = to_onnx_graph(fitted_transformer)

All 105 transformer classes are covered by ONNX converters.
