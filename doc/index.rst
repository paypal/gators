.. gators documentation

******
Gators
******

**Gators** is a machine learning library initially developed by the Simility a PayPal service Data Team. While data pre-processing and machine learning models are usually developed in Python, the pre-processing aspect is usually replaced by faster compiled programming languages in the production environment. This change of programming language is an added complexity to the model deployment process but is usually required to cope with the large number of queries per second that can be observed.

The goal of **Gators** is to be able to manage both model building and model serving using only Python, a language that data scientists are generally familiar with. **Gators** is built on top of Pandas, Koalas, NumPy and Cython. Pandas and Koalas are used for model building, while NumPy and Cython are used to speed up the model predictions in real-time. **Gators** was originally built for fraud modelling but can be generalized to many other modelling domains other than binary classification problems.

**Gators** helps to streamline the model building and productionization processes. The model building part can be done using the Pandas library for datasets held in memory, or Koalas for big data. On the model serving side, the pre-processing is carried out directly with Python, using NumPy and Cython. As a result, the speed-up using both NumPy and Cython for pre-processing is around 100 compared to standard Python code. Additionally, the per-sample response time becomes similar to other compiled languages (microsecond scale).

In summary, **Gators** is a package for handling model building, model deployment, and fast real-time pre-processing for a large number of QPS using only Python.

.. toctree::
    :maxdepth: 2

    about_gators/index
    getting_started/index
    user_guide/index
    reference/index
    benchmarking/index

