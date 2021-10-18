
.. _api.imputers:

********
Imputers
********

Four different types of imputers are available depending on the variable datatype,
namely: numerical, integer, float, and categorical (string or object).
 

.. note::

    * *NumericsImputer* imputes numerical variables.

    * *FloatImputer* imputes only numerical variables satisfying the condition:
      
         x != x.round().

    * *IntImputer* imputes only numerical variables satisfying the condition:
    
         x == x.round()

    * *ObjectImputer* imputes only categorical variables.


Base Imputer
############
.. currentmodule:: gators.imputers

.. autosummary::
   :toctree: api/

   _BaseImputer

Numerical Imputers
##################
.. currentmodule:: gators.imputers

.. autosummary::
   :toctree: api/

   NumericsImputer
   IntImputer
   FloatImputer

Categorical Imputer
###################

.. autosummary::
   :toctree: api/

   ObjectImputer

