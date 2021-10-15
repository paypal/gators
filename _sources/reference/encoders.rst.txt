
********
Encoders
********
.. _api.encoders:

The Encoders transform the categorical columns into numerical columns.

.. note::
    The Encoders transform the data inplace. The output of an encoder is then a numerical dataframe or array.
    Before calling an encoder, all the transformations on categorical columns should be done and the datetime columns should be dropped.

.. currentmodule:: gators.encoders

.. autosummary::
   :toctree: api/

BaseEncoder
###########

.. autosummary::
   :toctree: api/

   _BaseEncoder

Unsupervised Encoders
#####################

.. autosummary::
   :toctree: api/

   OrdinalEncoder
   OneHotEncoder

Binary Encoders
###############

.. autosummary::
   :toctree: api/

    WOEEncoder
    TargetEncoder

Multi-Class Encoder
###################

.. note::
    The *MultiClassEncoder* takes as input a binary encoder and apply it
    to each class. For example, if a n-class classification is composed of
    *c* categorical columns, the number encoded columns with be *n x c*.

.. autosummary::
   :toctree: api/

   MultiClassEncoder

Regression Encoder
##################

.. note::

    The *RegressionEncoder* takes as input a binary encoder and a discretizer.
    First, the discretizer transform the continuous target values into categorical values.
    Second, the MultiClassEncoder is applied to the data using the transformed target values.

.. autosummary::
   :toctree: api/

   RegressionEncoder

