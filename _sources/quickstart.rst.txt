Quick Start
===========

This guide will get you started with Gators in minutes.

Basic Example
-------------

Here's a simple example showing the core Gators workflow:

.. code-block:: python

    import polars as pl
    from gators.data_cleaning import DropHighNaNRatio, VarianceFilter
    from gators.encoders import OneHotEncoder
    from gators.imputers import NumericImputer
    from gators.scalers import StandardScaler
    from gators.pipeline import Pipeline

    # Load your data
    X = pl.read_csv("data.csv")

    # Build a preprocessing pipeline
    pipeline = Pipeline(steps=[
        ('drop_nan', DropHighNaNRatio(max_ratio=0.5)),   # drop columns with >50% nulls
        ('impute',   NumericImputer(strategy='median')), # fill numeric nulls with column median
        ('variance', VarianceFilter(min_var=0.01)),      # remove near-zero-variance columns
        ('encode',   OneHotEncoder()),                   # one-hot encode all string/categorical columns
        ('scale',    StandardScaler()),                  # z-score standardize numeric columns
    ])

    # Fit and transform in one step
    X_processed = pipeline.fit_transform(X)

    # Or fit and transform separately (e.g. train / test split)
    pipeline.fit(X_train)
    X_train_processed = pipeline.transform(X_train)
    X_test_processed  = pipeline.transform(X_test)

Understanding the API
---------------------

All Gators transformers follow the sklearn-style API:

**fit(X, y=None)**
    Learn parameters from the data (e.g., mean for imputation, categories for encoding).
    Supervised transformers (e.g. :class:`~gators.encoders.WOEEncoder`) also accept ``y``.

**transform(X)**
    Apply the learned transformation to ``X``.

**fit_transform(X, y=None)**
    Convenience method that calls ``fit()`` then ``transform()``.

Example: Data Cleaning
----------------------

.. code-block:: python

    from gators.data_cleaning import (
        DropHighNaNRatio,
        DropConstantColumns,
        DropNearConstantColumns,
        VarianceFilter,
        CorrelationFilter,
        RoundSignificantDigits,
    )

    # Drop columns with more than 50% missing values
    X = DropHighNaNRatio(max_ratio=0.5).fit_transform(X)

    # Drop constant and near-constant columns
    X = DropConstantColumns().fit_transform(X)
    X = DropNearConstantColumns(max_ratio=0.99).fit_transform(X)

    # Remove low-variance numeric features
    X = VarianceFilter(min_var=0.01).fit_transform(X)

    # Round to 3 significant figures for cleaner downstream processing
    X = RoundSignificantDigits(n_digits=3).fit_transform(X)

    # Remove highly correlated features, keeping the more important one
    importance = {"feature_a": 0.8, "feature_b": 0.6, "feature_c": 0.9}
    X = CorrelationFilter(max_corr=0.95).fit_transform(X)

Example: Missing Value Imputation
----------------------------------

.. code-block:: python

    from gators.imputers import NumericImputer, StringImputer, BooleanImputer

    # Impute numeric columns with column median
    X = NumericImputer(strategy='median').fit_transform(X)

    # Impute string columns with most frequent value
    X = StringImputer(strategy='most_frequent').fit_transform(X)

    # Impute boolean columns with False
    X = BooleanImputer(strategy='constant', value=False).fit_transform(X)

Example: Encoding
-----------------

.. code-block:: python

    from gators.encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder, WOEEncoder

    # One-hot encode all string/categorical columns
    X = OneHotEncoder().fit_transform(X)

    # Ordinal encode a subset of columns
    X = OrdinalEncoder(subset=['color', 'size']).fit_transform(X)

    # Target mean encoding (supervised — requires y)
    X = TargetEncoder(subset=['category_col']).fit_transform(X, y=target)

    # Weight of Evidence encoding (supervised — requires binary y)
    X = WOEEncoder(subset=['category_col']).fit_transform(X, y=binary_target)

Example: Feature Generation
----------------------------

.. code-block:: python

    from gators.feature_generation import (
        PolynomialFeatures,
        RatioFeatures,
        MathFeatures,
        GroupStatisticsFeatures,
    )
    from gators.feature_generation_dt import OrdinalFeatures, CyclicFeatures
    from gators.feature_generation_str import Length, NGram

    # Polynomial and interaction features (degree 2)
    X = PolynomialFeatures(subset=['amount', 'balance'], degree=2).fit_transform(X)

    # Ratio features: amount / balance
    X = RatioFeatures(
        numerator_columns=['amount'],
        denominator_columns=['balance'],
        new_column_names=['amount_to_balance_ratio'],
    ).fit_transform(X)

    # Group statistics: mean of 'amount' per 'merchant_category'
    X = GroupStatisticsFeatures(
        group_column='merchant_category',
        subset=['amount'],
        func='mean',
    ).fit_transform(X)

    # Datetime: extract year, month, day of week
    X = OrdinalFeatures(subset=['transaction_ts'], components=['year', 'month', 'day_of_week']).fit_transform(X)

    # Datetime: cyclical encoding for hour-of-day
    X = CyclicFeatures(subset=['transaction_ts'], components=['hour']).fit_transform(X)

    # String: length of description field
    X = Length(subset=['description']).fit_transform(X)

    # String: 2-gram features from category name
    X = NGram(subset=['category_name'], n=2).fit_transform(X)

Example: Scalers and Clippers
------------------------------

.. code-block:: python

    from gators.scalers import StandardScaler, RobustScaler, MinmaxScaler
    from gators.clippers import IQRClipper, QuantileClipper

    # Clip extreme values beyond 1st–99th percentiles before scaling
    X = QuantileClipper(lower_quantile=0.01, upper_quantile=0.99).fit_transform(X)

    # Robust scaling (unaffected by remaining outliers)
    X = RobustScaler().fit_transform(X)

Example: Complete Production Pipeline
--------------------------------------

.. code-block:: python

    import polars as pl
    from gators.pipeline import Pipeline
    from gators.data_cleaning import DropHighNaNRatio, DropConstantColumns, VarianceFilter
    from gators.imputers import NumericImputer, StringImputer
    from gators.encoders import WOEEncoder
    from gators.feature_generation import PolynomialFeatures
    from gators.scalers import StandardScaler

    pipeline = Pipeline(steps=[
        ('drop_nan',    DropHighNaNRatio(max_ratio=0.5)),
        ('drop_const',  DropConstantColumns()),
        ('variance',    VarianceFilter(min_var=0.01)),
        ('impute_num',  NumericImputer(strategy='median')),
        ('impute_str',  StringImputer(strategy='most_frequent')),
        ('polynomial',  PolynomialFeatures(subset=['amount', 'balance'], degree=2)),
        ('encode',      WOEEncoder()),
        ('scale',       StandardScaler()),
    ])

    # Fit on training data
    pipeline.fit(X_train, y=y_train)

    # Transform any dataset
    X_train_processed = pipeline.transform(X_train)
    X_test_processed  = pipeline.transform(X_test)

    # Export the fitted pipeline to ONNX for production inference
    from gators.onnx_converters import pipeline_to_onnx
    import onnxruntime as ort

    onnx_model = pipeline_to_onnx(pipeline)

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    # run sess.run(...) on any ONNX-compatible runtime

