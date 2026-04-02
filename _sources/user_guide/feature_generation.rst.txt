Feature Generation
==================

Gators provides 29 feature generators across numeric, string, and datetime domains.

Numeric Features
----------------

Polynomial Features
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation import PolynomialFeatures

    poly = PolynomialFeatures(
        columns=['feature1', 'feature2'],
        degree=2,
        interaction_only=False
    )
    X =  poly.fit_transform(X)

Ratio Features
~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation import RatioFeatures

    ratios = RatioFeatures(
        column_pairs=[('numerator', 'denominator')]
    )
    X =  ratios.fit_transform(X)

Math Features
~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation import MathFeatures

    math_features = MathFeatures(
        column_pairs=[('col1', 'col2')],
        operators=['+', '-', '*', '/']
    )
    X =  math_features.fit_transform(X)

Threshold Features
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation import ThresholXeatures

    threshold = ThresholXeatures(
        columns=['amount'],
        thresholds={'amount': 1000},
        operators='gte'
    )
    X =  threshold.fit_transform(X)

String Features
---------------

Length
~~~~~~

.. code-block:: python

    from gators.feature_generation_str import Length

    length = Length(columns=['text_column'])
    X =  length.fit_transform(X)

Contains
~~~~~~~~

.. code-block:: python

    from gators.feature_generation_str import Contains

    contains = Contains(
        contains_dict={'email': '@', 'phone': '-'}
    )
    X =  contains.fit_transform(X)

Pattern Detector
~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation_str import PatternDetector

    pattern = PatternDetector(
        columns=['text'],
        patterns={'has_number': r'\\d+'}
    )
    X =  pattern.fit_transform(X)

DateTime Features
-----------------

Datetime Ordinal Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation_dt import DatetimeOrdinalFeatures

    dt_features = DatetimeOrdinalFeatures(
        columns=['timestamp'],
        features=['year', 'month', 'day', 'hour', 'dayofweek']
    )
    X =  dt_features.fit_transform(X)

Datetime Cyclic Features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation_dt import DatetimeCyclicFeatures

    cyclic = DatetimeCyclicFeatures(
        columns=['timestamp'],
        features=['month', 'hour']
    )
    X =  cyclic.fit_transform(X)

Datetime Diff Features
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation_dt import DatetimeDiffFeatures

    diff = DatetimeDiffFeatures(
        column_pairs=[('end_time', 'start_time')],
        units=['days', 'hours']
    )
    X =  diff.fit_transform(X)

Holiday Features
~~~~~~~~~~~~~~~~

.. code-block:: python

    from gators.feature_generation_dt import HolidayFeatures

    holidays = HolidayFeatures(
        columns=['date'],
        country='US'
    )
    X =  holidays.fit_transform(X)

Best Practices
--------------

1. **Start simple**: Begin with basic features, add complexity as needed
2. **Domain knowledge**: Use business understanding to guide feature creation
3. **Feature selection**: Not all generated features will be useful
4. **Monitor cardinality**: Be careful with high-cardinality string features
