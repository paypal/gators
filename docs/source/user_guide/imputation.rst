Missing Value Imputation
========================

Gators provides 4 sophisticated imputers for handling missing data.

Numeric Imputer
---------------

Impute numeric columns with various strategies:

.. code-block:: python

    from gators.imputers import NumericImputer

    # Mean imputation
    imputer = NumericImputer(strategy='mean')
    X =  imputer.fit_transform(X)

    # Median imputation (robust to outliers)
    imputer = NumericImputer(strategy='median')
    X =  imputer.fit_transform(X)

    # Mode imputation
    imputer = NumericImputer(strategy='mode')
    X =  imputer.fit_transform(X)

    # Constant value imputation
    imputer = NumericImputer(strategy='constant', fill_value=0)
    X =  imputer.fit_transform(X)

String Imputer
--------------

Impute string columns:

.. code-block:: python

    from gators.imputers import StringImputer

    # Mode imputation
    imputer = StringImputer(strategy='mode')
    X =  imputer.fit_transform(X)

    # Constant value imputation
    imputer = StringImputer(strategy='constant', fill_value='unknown')
    X =  imputer.fit_transform(X)

Boolean Imputer
---------------

Impute boolean columns:

.. code-block:: python

    from gators.imputers import BooleanImputer

    imputer = BooleanImputer(fill_value=False)
    X =  imputer.fit_transform(X)

GroupBy Imputer
---------------

Group-based imputation for more sophisticated strategies:

.. code-block:: python

    from gators.imputers import GroupByImputer

    # Impute based on group statistics
    imputer = GroupByImputer(
        group_by=['category'],
        strategy='mean',
        columns=['value']
    )
    X =  imputer.fit_transform(X)

Best Practices
--------------

1. **Understand your data**: Know why values are missing
2. **Choose appropriate strategy**: 
   - Use median for skewed distributions
   - Use mean for normal distributions
   - Use mode for categorical data
3. **Group-based when possible**: GroupBy imputation preserves relationships
4. **Document strategy**: Keep track of imputation methods used
5. **Consider indicators**: Create missing value indicator features

Complete Imputation Pipeline
-----------------------------

.. code-block:: python

    from gators.pipeline import Pipeline
    from gators.imputers import (
        NumericImputer,
        StringImputer,
        BooleanImputer
    )

    imputation_pipeline = Pipeline([
        ('impute_numeric', NumericImputer(strategy='median')),
        ('impute_string', StringImputer(strategy='mode')),
        ('impute_bool', BooleanImputer(fill_value=False))
    ])

    X =  imputation_pipeline.fit_transform(X)
