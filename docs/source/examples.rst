Examples
========

This section contains end-to-end examples demonstrating how to use Gators for real-world machine learning tasks.

.. toctree::
   :maxdepth: 2
   :hidden:

   examples/titanic_survival_prediction
   examples/house_price_prediction
   examples/sf_crime_classification
   examples/fraud_detection
   examples/benchmark_encoders
   examples/benchmark_encoders_feature_engine
   examples/benchmark_imputers
   examples/benchmark_imputers_feature_engine
   examples/benchmark_scalers
   examples/benchmark_scalers_feature_engine
   examples/benchmark_datetime_features_feature_engine

Overview
--------

Each example notebook demonstrates a complete ML workflow using Gators transformers:

* Data loading and exploration
* Feature engineering pipeline construction
* Model training and evaluation
* Performance comparison with traditional methods


Titanic Survival Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary classification using advanced feature engineering including string parsing, mathematical features,
and rare category encoding.

:doc:`View Notebook <examples/titanic_survival_prediction>`

House Price Prediction
~~~~~~~~~~~~~~~~~~~~~~~

Regression task demonstrating numeric feature engineering, scaling, and handling of mixed data types.

:doc:`View Notebook <examples/house_price_prediction>`

San Francisco Crime Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-class classification with datetime feature engineering, text processing, and geographic features.

:doc:`View Notebook <examples/sf_crime_classification>`

Fraud Detection
~~~~~~~~~~~~~~~

Imbalanced classification with advanced feature generation, group-based statistics, and model evaluation.

:doc:`View Notebook <examples/fraud_detection>`

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

Benchmark Gators vs Sklearn Encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators encoders against sklearn encoders across different dataset sizes,
demonstrating Polars' multi-core processing advantages.

:doc:`View Notebook <examples/benchmark_encoders>`

Benchmark Gators vs Feature-Engine Encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators encoders against feature-engine encoders, showcasing speedup
across OneHot, Ordinal, Count, and RareCategory encoding transformers.

:doc:`View Notebook <examples/benchmark_encoders_feature_engine>`

Benchmark Gators vs Sklearn Imputers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators imputers against sklearn imputers across different dataset sizes,
comparing NumericImputer, StringImputer, and BooleanImputer with multiple strategies.

:doc:`View Notebook <examples/benchmark_imputers>`

Benchmark Gators vs Feature-Engine Imputers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators imputers against feature-engine imputers, showcasing speedup
across mean, median, constant, and categorical imputation strategies.

:doc:`View Notebook <examples/benchmark_imputers_feature_engine>`

Benchmark Gators vs Sklearn Scalers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators scalers against sklearn scalers across different dataset sizes,
comparing StandardScaler, MinMaxScaler, BoxCox, and YeoJohnson transformations.

:doc:`View Notebook <examples/benchmark_scalers>`

Benchmark Gators vs Feature-Engine Scalers/Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators scalers against feature-engine transformers and sklearn scalers,
showcasing speedup across BoxCox, YeoJohnson power transformations and traditional scalers.

:doc:`View Notebook <examples/benchmark_scalers_feature_engine>`

Benchmark Gators vs Feature-Engine Datetime Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance comparison of Gators datetime feature generators against feature-engine DatetimeFeatures,
showcasing speedup across ordinal features (year, month, day, etc.), day of week, and quarter extraction.

:doc:`View Notebook <examples/benchmark_datetime_features_feature_engine>`
