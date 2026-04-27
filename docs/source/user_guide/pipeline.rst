Pipeline
========

The Pipeline class allows you to chain multiple transformers together in a single, reusable object.

Basic Usage
-----------

.. code-block:: python

    from gators.pipeline import Pipeline
    from gators.data_cleaning import DropHighNaNRatio
    from gators.imputers import NumericImputer
    from gators.encoders import OneHotEncoder
    from gators.scalers import StandardScaler

    # Create pipeline
    pipeline = Pipeline([
        ('drop_nan', DropHighNaNRatio(threshold=0.5)),
        ('impute', NumericImputer(strategy='median')),
        ('encode', OneHotEncoder()),
        ('scale', StandardScaler())
    ])

    # Fit and transform
    X_transformed = pipeline.fit_transform(X)

Pipeline Benefits
-----------------

1. **Code Organization**: Encapsulate entire preprocessing workflow
2. **Reproducibility**: Ensure consistent transformations
3. **Production Deployment**: Deploy entire pipeline as single object
4. **Parameter Tuning**: Easy to tune hyperparameters across all steps
5. **Serialization**: Save and load complete pipelines

Accessing Pipeline Steps
-------------------------

.. code-block:: python

    # Access specific step
    imputer = pipeline.named_steps['impute']

    # Get all steps
    steps = pipeline.steps

    # Iterate through steps
    for name, transformer in pipeline.named_steps.items():
        print(f"{name}: {transformer}")

Fitting and Transforming
-------------------------

.. code-block:: python

    # Option 1: Fit and transform together
    X_transformed = pipeline.fit_transform(X)

    # Option 2: Fit then transform
    pipeline.fit(X_train)
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

Production-Ready Pipeline
--------------------------

.. code-block:: python

    from gators.pipeline import Pipeline
    from gators.data_cleaning import (
        DropHighNaNRatio,
        VarianceFilter,
        CorrelationFilter
    )
    from gators.imputers import NumericImputer, StringImputer
    from gators.feature_generation import PolynomialFeatures
    from gators.feature_generation_dt import DatetimeOrdinalFeatures
    from gators.encoders import TargetEncoder, OneHotEncoder
    from gators.scalers import StandardScaler

    # Define comprehensive pipeline
    ml_pipeline = Pipeline([
        # Data Cleaning
        ('drop_nan', DropHighNaNRatio(threshold=0.5)),
        ('variance', VarianceFilter(threshold=0.01)),
        ('correlation', CorrelationFilter(threshold=0.95)),
        
        # Feature Engineering
        ('datetime', DatetimeOrdinalFeatures(
            columns=['timestamp'],
            features=['year', 'month', 'day', 'dayofweek']
        )),
        ('polynomial', PolynomialFeatures(
            columns=['num1', 'num2'],
            degree=2
        )),
        
        # Imputation
        ('impute_numeric', NumericImputer(strategy='median')),
        ('impute_string', StringImputer(strategy='mode')),
        
        # Encoding
        ('target_encode', TargetEncoder(
            columns=['high_card_cat']
        )),
        ('onehot_encode', OneHotEncoder()),
        
        # Scaling
        ('scale', StandardScaler())
    ])

    # Train
    ml_pipeline.fit(train_X, y=train_target)
    X_train = ml_pipeline.transform(train_X)

    # Test
    X_test = ml_pipeline.transform(test_X)

    # Production
    X_prod = ml_pipeline.transform(prod_X)

Serialization
-------------

Save and load pipelines for deployment:

.. code-block:: python

    import pickle

    # Save pipeline
    with open('preprocessing_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    # Load pipeline
    with open('preprocessing_pipeline.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)

    # Use loaded pipeline
    X_transformed = loaded_pipeline.transform(new_data)

Best Practices
--------------

1. **Name your steps**: Use descriptive names for debugging
2. **Order matters**: Clean → Impute → Engineer → Encode → Scale
3. **Fit on training only**: Prevent data leakage
4. **Document parameters**: Keep track of hyperparameters used
5. **Version control**: Save pipeline versions alongside models
6. **Test thoroughly**: Validate pipeline on multiple datasets

Integration with sklearn
-------------------------

Gators pipelines work seamlessly with scikit-learn:

.. code-block:: python

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from gators.pipeline import Pipeline as GatorsPipeline

    # Combine Gators preprocessing with sklearn model
    full_pipeline = SklearnPipeline([
        ('preprocessing', GatorsPipeline([
            ('impute', NumericImputer(strategy='median')),
            ('encode', OneHotEncoder()),
            ('scale', StandardScaler())
        ])),
        ('model', LogisticRegression())
    ])

    # Train end-to-end
    full_pipeline.fit(X_train, y_train)
    predictions = full_pipeline.predict(X_test)
