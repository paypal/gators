************
About Gators
************

**Gators** was created to help data scientists to:

* Perform in-memory and out-of-core memory data pre-processing for model building.
* Fast real-time pre-processing and model scoring.

######################
History of development
######################

In 2018, **Gators** development began at Simility and had been open sourced in 2021,

========
Timeline
========
* **2018**: Development of **Gators** started.
* **2020**: Koalas and Cython packages are added to tackle out-of-core memory datasets and fast real-time pre-processing. 
* **2021**: **Gators** becomes open source.

##################
Library Highlights
##################

* Data pre-processing can be done for both in-memory and out-of-memory datasets using the same interface.
* Using Cython, the real-time data pre-processing is carried out on NumPy arrays with compiled C-code leading to fast response times, similar to compiled languages.

##########
Our Vision
##########

A world where data scientists can develop and push their models in production using only Python, even when there are a large number of queries per second (QPS).

###################################
Python packages leveraged in gators
###################################

.. centered:: "If I have seen further it is by standing on the shoulders of Giants."

.. centered:: Sir Isaac Newton

**gators** uses a variety of libraries internally, at each step of the model building process.

Below is the list of libraries used.

===================
Data pre-processing
===================

.. image:: ../_static/pandas_logo.png

The most well known package for data analysis is used for data pre-processing during the model building phase. This package should be used as long as the data can fit in memory.

.. image:: ../_static/koalas_logo.png
    :width: 170 px

koalas has been chosen to replace pandas if the data does not fit in memory. The main advantage of koalas compared to other big data packages such as PySpark and Dask, is the fact that the syntax is close to pandas.

.. image:: ../_static/numpy_logo.png
    :width: 170 px

NumPy is used in the production environment when the pre-processing needs to be as fast as possible.

.. image:: ../_static/cython_logo.jpeg
    :width: 170 px

In the production environment, the pre-processing with be done by pre-compile Cython code on NumPy arrays.

==============
Model building
==============

.. image:: ../_static/sklearn_logo.png
    :width: 170 px

The most well known package for model building is used for cross-validation and model evaluation.

.. image:: ../_static/hyperopt_logo.png
    :width: 170 px

This package is used for hyperparameter tuning. The three algorithms currently available to perform hyperparameter tuning are:

* Random Search
* Tree of Parzen Estimators (TPE)
* Adaptative TPE

.. image:: ../_static/xgboost_logo.png
    :width: 170 px

Decision tree-based package used for model building. XGBoost algorithm applies level-wise tree growth.

.. image:: ../_static/lightgbm_logo.png
    :width: 170 px

Decision tree-based package used for model building. LightGBM algorithm applies leaf-wise tree growth.

.. image:: ../_static/treelite_logo.png
    :width: 170 px

Treelite is used to compile the trained models in C before being deployed in production,
and treelite-runtime is used for real-time model scoring.

