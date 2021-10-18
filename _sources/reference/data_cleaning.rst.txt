
.. _api.data_cleaning:

*************
Data Cleaning
*************

.. currentmodule:: gators.data_cleaning

These transformers can be used to reduce the number of columns
during the feature selection step.

Base data_cleaning transformer
##############################

.. autosummary::
   :toctree: api/
   
    _BaseDataCleaning 

Off-line data cleaning
###################### 

.. autosummary::
   :toctree: api/

    DropHighCardinality    
    DropHighNaNRatio     
    DropLowCardinality

Realtime data cleaning
###################### 

.. autosummary::
   :toctree: api/

    DropColumns 
    DropDatatypeColumns     
    KeepColumns
    Replace    

