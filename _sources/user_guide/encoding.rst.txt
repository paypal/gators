Categorical Encoding
====================

Gators provides 9 advanced encoding techniques for categorical variables.

OneHot Encoding
---------------

Classic one-hot encoding for nominal categories:

.. code-block:: python

    from gators.encoders import OneHotEncoder

    encoder = OneHotEncoder(columns=['category'])
    X =  encoder.fit_transform(X)

Target Encoding
---------------

Mean target encoding for supervised learning:

.. code-block:: python

    from gators.encoders import TargetEncoder

    encoder = TargetEncoder(columns=['category'])
    X =  encoder.fit_transform(X, y=target)

WOE Encoding
------------

Weight of Evidence encoding:

.. code-block:: python

    from gators.encoders import WOEEncoder

    encoder = WOEEncoder(columns=['category'])
    X =  encoder.fit_transform(X, y=target)

Ordinal Encoding
----------------

Order-based encoding for ordinal categories:

.. code-block:: python

    from gators.encoders import OrdinalEncoder

    encoder = OrdinalEncoder(
        columns=['size'],
        categories=['small', 'medium', 'large']
    )
    X =  encoder.fit_transform(X)

Count Encoding
--------------

Frequency-based encoding:

.. code-block:: python

    from gators.encoders import CountEncoder

    encoder = CountEncoder(columns=['category'])
    X =  encoder.fit_transform(X)

Binary Encoding
---------------

.. code-block:: python

    from gators.encoders import BinaryEncoder

    encoder = BinaryEncoder(columns=['category'])
    X =  encoder.fit_transform(X)

CatBoost Encoding
-----------------

.. code-block:: python

    from gators.encoders import CatBoostEncoder

    encoder = CatBoostEncoder(columns=['category'])
    X =  encoder.fit_transform(X, y=target)

Leave One Out Encoding
----------------------

.. code-block:: python

    from gators.encoders import LeaveOneOutEncoder

    encoder = LeaveOneOutEncoder(columns=['category'])
    X =  encoder.fit_transform(X, y=target)

Rare Category Encoding
----------------------

Handle rare categories intelligently:

.. code-block:: python

    from gators.encoders import RareCategoryEncoder

    encoder = RareCategoryEncoder(
        columns=['category'],
        threshold=0.01  # Categories with <1% frequency
    )
    X =  encoder.fit_transform(X)

Best Practices
--------------

1. **Choose appropriate encoding**: Use target encoding for high-cardinality features
2. **Handle unseen categories**: Set ``handle_unknown`` parameter
3. **Regularization**: Use smoothing for target-based encoders
4. **Train/test consistency**: Always fit on training data only
