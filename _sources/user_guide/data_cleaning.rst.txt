Data Cleaning
=============

Gators provides 13 powerful transformers for data cleaning and quality control.

Column Operations
-----------------

Rename Columns
~~~~~~~~~~~~~~

.. code-block:: python

    from gators.data_cleaning import RenameColumns

    renamer = RenameColumns({'old_name': 'new_name'})
    X =  renamer.transform(X)

Cast Columns
~~~~~~~~~~~~

.. code-block:: python

    from gators.data_cleaning import CastColumns

    caster = CastColumns({'column': pl.Float64})
    X =  caster.transform(X)

Drop Columns
~~~~~~~~~~~~

.. code-block:: python

    from gators.data_cleaning import DropColumns

    dropper = DropColumns(columns=['col1', 'col2'])
    X =  dropper.transform(X)

Quality Filters
---------------

Drop High NaN Ratio
~~~~~~~~~~~~~~~~~~~

Remove columns with too many missing values:

.. code-block:: python

    from gators.data_cleaning import DropHighNaNRatio

    # Drop columns with >50% missing values
    cleaner = DropHighNaNRatio(threshold=0.5)
    X =  cleaner.fit_transform(X)

Variance Filter
~~~~~~~~~~~~~~~

Remove low-variance columns:

.. code-block:: python

    from gators.data_cleaning import VarianceFilter

    # Drop columns with variance < 0.01
    var_filter = VarianceFilter(threshold=0.01)
    X =  var_filter.fit_transform(X)

Correlation Filter
~~~~~~~~~~~~~~~~~~

Remove highly correlated features:

.. code-block:: python

    from gators.data_cleaning import CorrelationFilter

    # Drop one of each pair with correlation > 0.95
    corr_filter = CorrelationFilter(threshold=0.95)
    X =  corr_filter.fit_transform(X)

Outlier Detection
-----------------

.. code-block:: python

    from gators.data_cleaning import OutlierFilter

    # Remove rows with outliers using IQR method
    outlier_filter = OutlierFilter(
        columns=['feature1', 'feature2'],
        method='iqr',
        threshold=1.5
    )
    X =  outlier_filter.fit_transform(X)

Deduplication
-------------

Duplicate Columns Remover
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.data_cleaning import DuplicateColumnsRemover

    deduper = DuplicateColumnsRemover()
    X =  deduper.fit_transform(X)

Duplicate Rows Remover
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.data_cleaning import DuplicateRowsRemover

    row_deduper = DuplicateRowsRemover()
    X =  row_deduper.transform(X)

Constant Columns Remover
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.data_cleaning import ConstantColumnsRemover

    constant_remover = ConstantColumnsRemover()
    X =  constant_remover.fit_transform(X)

Best Practices
--------------

1. **Order matters**: Apply quality filters before encoding
2. **Fit on training data only**: Use ``.fit()`` on training, then ``.transform()`` on test
3. **Document thresholds**: Keep track of threshold values used
4. **Monitor removed features**: Log which columns are dropped

Complete Cleaning Pipeline
---------------------------

.. code-block:: python

    from gators.pipeline import Pipeline
    from gators.data_cleaning import (
        DropHighNaNRatio,
        VarianceFilter,
        CorrelationFilter,
        DuplicateColumnsRemover,
        ConstantColumnsRemover
    )

    cleaning_pipeline = Pipeline([
        ('drop_nan', DropHighNaNRatio(threshold=0.5)),
        ('constants', ConstantColumnsRemover()),
        ('duplicates', DuplicateColumnsRemover()),
        ('variance', VarianceFilter(threshold=0.01)),
        ('correlation', CorrelationFilter(threshold=0.95))
    ])

    X_clean = cleaning_pipeline.fit_transform(X)
