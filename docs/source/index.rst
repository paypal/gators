
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
* 🎯 **Comprehensive**: 105 preprocessing transformers across 11 categories
* 🔗 **Pipeline Support**: Chain transformers seamlessly with the Pipeline class
* 📤 **ONNX Export**: Export any fitted pipeline to ONNX for low-latency inference
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
    pipeline = Pipeline(steps=[
        ('drop_nan', DropHighNaNRatio(max_ratio=0.5)),  # drop columns with >50% missing values
        ('impute', NumericImputer(strategy='median')),  # impute missing values with median
        ('variance', VarianceFilter(min_var=0.01)),     # remove numerical columns with a variance < 0.01
        ('encode', OneHotEncoder()),  # one-hot encode categorical variables
        ('scale', StandardScaler())  # standardize numerical features
    ])

    # Fit and transform
    X_processed = pipeline.fit_transform(X)

    # Deploy the same pipeline in production!

What Can Gators Do?
===================

**70+ transformers across 8 categories:**

* 🧹 :doc:`Data Cleaning <api/data_cleaning>` - Quality filters, deduplication, rounding, and type casting (16)
* ✂️ :doc:`Clippers <api/clippers>` - Custom, Gaussian, IQR, MAD, and Quantile outlier clipping (5)
* 🧩 :doc:`Encoders <api/encoders>` - OneHot, Target, WOE, CatBoost, Binary, Hash, and more (10)
* 🎯 :doc:`Numeric Features <api/feature_generation>` - Polynomial, ratio, Fourier, aggregation, rule-based (20)
* 📝 :doc:`String Features <api/feature_generation_str>` - Length, patterns, n-grams, TF-IDF, regex extraction (17)
* 📅 :doc:`DateTime Features <api/feature_generation_dt>` - Cyclical encoding, holidays, business hours, time windows (8)
* 🔄 :doc:`Imputation <api/imputers>` - Numeric, string, boolean, KNN, iterative, and group-based strategies (6)
* 📊 :doc:`Discretization <api/discretizers>` - Equal-width, quantile, tree-based binning, and more (7)
* ⚖️ :doc:`Scalers <api/scalers>` - Standard, min-max, robust, Box-Cox, Yeo-Johnson, and more (9)
* ✨ :doc:`Feature Selection <api/feature_selection>` - Correlation, stability, IV, mutual information, permutation (6)
* 🔗 :doc:`Pipeline <api/pipeline>` - Chain transformers seamlessly (1)
* 📤 **ONNX Export** - Export any fitted pipeline to ONNX for language-agnostic inference
  

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
