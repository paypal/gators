Quick Start
===========

This guide will get you started with Gators in minutes.

Basic Example
-------------

Here's a simple example showing the core Gators workflow:

.. code-block:: python

    import polars as pl
    from gators.data_cleaning import DropHighNaNRatio
    from gators.encoders import OneHotEncoder
    from gators.imputers import NumericImputer
    from gators.scalers import StandardScaler
    from gators.pipeline import Pipeline

    # Load your data
    X =  pl.read_csv("data.csv")

    # Build a preprocessing pipeline
    pipeline = Pipeline([
        ('drop_nan', DropHighNaNRatio(threshold=0.5)),
        ('impute', NumericImputer(strategy='median')),
        ('encode', OneHotEncoder()),
        ('scale', StandardScaler())
    ])

    # Fit and transform in one step
    X_processed = pipeline.fit_transform(X)

    # Or fit and transform separately
    pipeline.fit(X)
    X_processed = pipeline.transform(X)

Understanding the API
---------------------

All Gators transformers follow the sklearn-style API:

**fit(X)**
    Learn parameters from the data (e.g., mean for imputation, categories for encoding)

**transform(X)**
    Apply the transformation using learned parameters

**fit_transform(X)**
    Convenience method that calls fit() then transform()

Example: Data Cleaning
----------------------

.. code-block:: python

    from gators.data_cleaning import (
        DropHighNaNRatio,
        VarianceFilter,
        CorrelationFilter
    )

    # Remove columns with >50% missing values
    drop_nan = DropHighNaNRatio(threshold=0.5)
    X =  drop_nan.fit_transform(X)

    # Remove low-variance features
    var_filter = VarianceFilter(threshold=0.01)
    X =  var_filter.fit_transform(X)

    # Remove highly correlated features
    corr_filter = CorrelationFilter(threshold=0.95)
    X =  corr_filter.fit_transform(X)

Example: Encoding
-----------------

.. code-block:: python

    from gators.encoders import (
        OneHotEncoder,
        TargetEncoder,
        OrdinalEncoder
    )

    # One-hot encoding
    ohe = OneHotEncoder(columns=['category_col'])
    X =  ohe.fit_transform(X)

    # Ordinal encoding
    ordinal = OrdinalEncoder(columns=['category_col'])
    X =  ordinal.fit_transform(X)

    # Target encoding (for supervised learning)
    target_encoder = TargetEncoder(columns=['category_col'])
    X =  target_encoder.fit_transform(X, y=target)

Example: Feature Generation
----------------------------

.. code-block:: python

    from gators.feature_generation import (
        PolynomialFeatures,
        RatioFeatures,
    )
    from gators.feature_generation_dt import DatetimeOrdinalFeatures

    # Create polynomial features
    poly = PolynomialFeatures(columns=['feature1', 'feature2'], degree=2)
    X =  poly.fit_transform(X)

    # Create ratio features
    ratios = RatioFeatures(column_pairs=[('numerator', 'denominator')])
    X =  ratios.fit_transform(X)

    # Extract datetime features
    dt_features = DatetimeOrdinalFeatures(
        columns=['timestamp'],
        features=['year', 'month', 'day', 'hour']
    )
    X =  dt_features.fit_transform(X)

Example: Complete Pipeline
---------------------------

Putting it all together in a production-ready pipeline:

.. code-block:: python

    from gators.pipeline import Pipeline
    from gators.data_cleaning import DropHighNaNRatio, VarianceFilter
    from gators.imputers import NumericImputer, StringImputer
    from gators.encoders import OneHotEncoder
    from gators.feature_generation import PolynomialFeatures
    from gators.scalers import StandardScaler

    # Define the complete pipeline
    pipeline = Pipeline([
        # Step 1: Clean data
        ('drop_high_nan', DropHighNaNRatio(threshold=0.5)),
        ('variance_filter', VarianceFilter(threshold=0.01)),
        
        # Step 2: Handle missing values
        ('impute_numeric', NumericImputer(strategy='median')),
        ('impute_string', StringImputer(strategy='mode')),
        
        # Step 3: Feature engineering
        ('polynomial', PolynomialFeatures(
            columns=['feature1', 'feature2'], 
            degree=2
        )),
        
        # Step 4: Encode categorical variables
        ('encode', OneHotEncoder()),
        
        # Step 5: Scale features
        ('scale', StandardScaler())
    ])

    # Fit on training data
    pipeline.fit(train_X)

    # Transform training and test data
    train_processed = pipeline.transform(train_X)
    test_processed = pipeline.transform(test_X)

    # Deploy the same pipeline in production!
    prod_data_processed = pipeline.transform(prod_data)
