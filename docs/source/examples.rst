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

Overview
--------

Each example notebook demonstrates a complete ML workflow using Gators transformers:

* Data loading and exploration
* Feature engineering pipeline construction
* Model training and evaluation
* Performance comparison with traditional methods
* End-to-end ONNX export of the fitted pipeline, chaining feature preprocessing, feature
  generation, and model scoring into a single portable inference graph


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
