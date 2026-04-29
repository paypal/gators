
.. image:: _static/GATORS_LOGO.png
   :align: center
   :alt: Gators Logo

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples

Gators is a **lightning-fast data preprocessing and feature engineering** library built on top of `Polars <https://pola.rs/>`_, 
designed to streamline your entire ML workflow from raw data to production-ready models. Leveraging Polars' 
blazing-fast multi-core processing.

Built by the PSP Data Team at PayPal, Gators makes data preprocessing and feature engineering both **faster** and **simpler**.

Key Features
============

* 🚀 **Lightning Fast**: Built on Polars for multi-core parallel processing
* 🔄 **Unified API**: Consistent sklearn-style ``.fit()`` and ``.transform()`` interface
* 📦 **Production Ready**: Deploy the same Python code from notebook to production
* 🎯 **Comprehensive**: 60+ preprocessing transformers covering every use case
* 🔗 **Pipeline Support**: Chain transformers seamlessly with the Pipeline class
* 🎓 **Easy to Learn**: If you know sklearn, you already know Gators

Quick Start
===========

.. code-block:: python

    import polars as pl
    from gators.data_cleaning import DropHighNaNRatio, VarianceFilter
    from gators.encoders import OneHotEncoder
    from gators.imputers import NumericImputer
    from gators.scalers import StandardScaler
    from gators.pipeline import Pipeline

    # Load your data
    X =  pl.read_csv("data.csv")

    # Build a preprocessing pipeline
    pipeline = Pipeline([
        ('drop_nan', DropHighNaNRatio(threshold=0.5)),  # drop columns with >50% missing values
        ('impute', NumericImputer(strategy='median')),  # impute missing values with median
        ('variance', VarianceFilter(threshold=0.01)),   # remove numerical columns with a variance < 0.01
        ('encode', OneHotEncoder()),  # one-hot encode categorical variables
        ('scale', StandardScaler())  # standardize numerical features
    ])

    # Fit and transform
    X_processed = pipeline.fit_transform(X)

    # Deploy the same pipeline in production!

What Can Gators Do?
===================

**70+ transformers across 8 categories:**

* 🧹 :doc:`Data Cleaning <api/data_cleaning>` - Quality filters, deduplication, and more
* ✂️ :doc:`Clippers <api/clippers>` - Custom min/max bounds, Gaussian, IQR, MAD, Quantile, and more
* 🧩 :doc:`Encoders <api/encoders>` - OneHot, Target, WOE, CatBoost, and more   
* 🎯 :doc:`Numeric Features <api/feature_generation>` - Polynomial, rule-based features, and more
* 📝 :doc:`String Features <api/feature_generation_str>` - Text properties, pattern detection, n-grams, and more
* 📅 :doc:`DateTime Features <api/feature_generation_dt>` - Temporal patterns, cyclical encoding, holidays, and more
* 🔄 :doc:`Imputation <api/imputers>` - Numeric, string, boolean, and group-based strategies
* 📊 :doc:`Discretization <api/discretizers>` - Equal-width, quantile, tree-based binning, and more
* ⚖️ :doc:`Scaling <api/scalers>` - Standard, min-max, Box-Cox, and more
* 🔗 :doc:`Pipeline <api/pipeline>` - Chain transformers seamlessly
  

Credits
-------

Developed by the PSP Data Team at PayPal.

**⚡ Built by data scientists, for data scientists**

Standing on the Shoulders of Giants
------------------------------------

    *"If I have seen further, it is by standing on the shoulders of giants."* — Isaac Newton

Gators builds upon the incredible work of the open-source community. We are deeply grateful to:

* **scikit-learn** (`scikit-learn.org <https://scikit-learn.org/>`_) - Inspired Gators' API design
* **feature-engine** (`feature-engine.trainindata.com <https://feature-engine.trainindata.com/>`_) - Inspired Gators' transformer patterns

Gators continues this tradition with Polars-powered performance.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
