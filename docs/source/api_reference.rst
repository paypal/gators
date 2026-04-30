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

Gators provides 75+ transformers across 11 categories, all with a consistent sklearn-compatible API.
Each transformer implements ``.fit()`` and ``.transform()`` methods and works seamlessly with Polars DataFrames.


Data Cleaning
~~~~~~~~~~~~~

Quality filters, variance detection, correlation removal, and data quality transformations.

:doc:`View Data Cleaning API <api/data_cleaning>`

Clippers
~~~~~~~~

Outlier detection and clipping strategies including Gaussian, IQR, MAD, and Quantile methods.

:doc:`View Clippers API <api/clippers>`

Encoders
~~~~~~~~

Categorical encoding methods: OneHot, Target, WOE, CatBoost, Ordinal, and more.

:doc:`View Encoders API <api/encoders>`

Feature Generation
~~~~~~~~~~~~~~~~~~

Create numeric features from existing columns: polynomial, ratios, aggregations, and custom transformations.

:doc:`View Feature Generation API <api/feature_generation>`

String Features
~~~~~~~~~~~~~~~

Extract information from text: length, patterns, n-grams, substring extraction, and text statistics.

:doc:`View String Features API <api/feature_generation_str>`

DateTime Features
~~~~~~~~~~~~~~~~~

Temporal feature engineering: cyclic encoding, holidays, business hours, time windows, and date differences.

:doc:`View DateTime Features API <api/feature_generation_dt>`

Imputers
~~~~~~~~

Handle missing values with mean, median, mode, constant, or group-based imputation strategies.

:doc:`View Imputers API <api/imputers>`

Discretizers
~~~~~~~~~~~~

Bin continuous variables using equal-width, quantile, k-means, tree-based, or custom strategies.

:doc:`View Discretizers API <api/discretizers>`

Scalers
~~~~~~~

Normalize and transform features: StandardScaler, MinMaxScaler, Box-Cox, Yeo-Johnson, and more.

:doc:`View Scalers API <api/scalers>`

Pipeline
~~~~~~~~

Chain multiple transformers together for streamlined preprocessing workflows.

:doc:`View Pipeline API <api/pipeline>`

Feature Selection
~~~~~~~~~~~~~~~~~

Select important features using Information Value, Feature Stability Index, and other metrics.

:doc:`View Feature Selection API <api/feature_selection>`
