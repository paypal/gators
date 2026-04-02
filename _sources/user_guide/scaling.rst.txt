Feature Scaling
===============

Gators provides 3 scalers for feature normalization.

Standard Scaler
---------------

Z-score normalization (mean=0, std=1):

.. code-block:: python

    from gators.scalers import StandardScaler

    scaler = StandardScaler()
    X =  scaler.fit_transform(X)

    # Optionally specify columns
    scaler = StandardScaler(columns=['feature1', 'feature2'])
    X =  scaler.fit_transform(X)

MinMax Scaler
-------------

Scale features to a fixed range [0, 1]:

.. code-block:: python

    from gators.scalers import MinmaxScaler

    scaler = MinmaxScaler()
    X =  scaler.fit_transform(X)

    # Custom range
    scaler = MinmaxScaler(feature_range=(0, 10))
    X =  scaler.fit_transform(X)

Yeo-Johnson Transformer
-----------------------

Power transformation for making data more Gaussian:

.. code-block:: python

    from gators.scalers import YeoJonhson

    transformer = YeoJonhson()
    X =  transformer.fit_transform(X)

When to Use Each Scaler
------------------------

**StandardScaler**
    - Best for: Most machine learning algorithms (SVM, neural networks, linear models)
    - When: Features have different units or scales
    - Not for: Tree-based models (they're scale-invariant)

**MinmaxScaler**
    - Best for: Neural networks, algorithms sensitive to feature ranges
    - When: Need bounded features in a specific range
    - Be careful: Sensitive to outliers

**YeoJonhson**
    - Best for: Making skewed distributions more Gaussian
    - When: Algorithms assume normality (linear regression, LDA)
    - Advantage: Handles both positive and negative values

Best Practices
--------------

1. **Fit on training data only**: Prevent data leakage

   .. code-block:: python

       scaler.fit(train_X)
       train_scaled = scaler.transform(train_X)
       test_scaled = scaler.transform(test_X)

2. **Scale after splitting**: Always split before scaling
3. **Tree models don't need scaling**: Random Forest, XGBoost, etc.
4. **Handle outliers first**: Outliers can skew scaling
5. **Scale after encoding**: One-hot encoded features don't need scaling

Complete Scaling Example
-------------------------

.. code-block:: python

    from gators.pipeline import Pipeline
    from gators.data_cleaning import OutlierFilter
    from gators.encoders import OneHotEncoder
    from gators.scalers import StandardScaler

    pipeline = Pipeline([
        # Remove outliers first
        ('outliers', OutlierFilter(method='iqr')),
        
        # Encode categorical variables
        ('encode', OneHotEncoder()),
        
        # Scale numeric features
        ('scale', StandardScaler())
    ])

    # Fit on training data
    pipeline.fit(train_X)
    
    # Transform both sets
    train_scaled = pipeline.transform(train_X)
    test_scaled = pipeline.transform(test_X)
