# gators

[![Build Status](https://img.shields.io/travis/dmlc/treelite.svg?label=build&logo=travis&branch=mainline)](https://travis-ci.org/dmlc/treelite)
[![Documentation Status](https://readthedocs.org/projects/treelite/badge/?version=latest)](http://treelite.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/scikit-learn-contrib/imbalanced-learn/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-learn-contrib/imbalanced-learn)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

<a href="https://paypal.github.io/gators/index.html" target="_blank">Documentation</a> | 
<a href="https://paypal.github.io/gators/index.html/getting_started/install.html" target="_blank">Installation</a>


![Gators logo](doc_data/GATORS_LOGO.png)

Gators is an in-house machine learning library developed by the Simility Data Team. While data pre-processing and machine learning models are developed in Python, the pre-processing aspect is replaced by faster compiled programming languages in the production environment. This change of programming language is an added complexity to the model deployment process but is required to cope with the large number of queries per second that can be observed.

The goal of Gators is to be able to manage both model building and model serving using only Python, a language that data scientists are generally familiar with. Gators is built on top of Pandas, Koalas, NumPy and Cython. Pandas and Koalas are used for the offline model building, while NumPy and Cython are used to speed-up the model predictions in real-time. Gators was originally built for fraud modelling but can be generalized to other modelling domains.

Gators helps to streamline the model building and productionization processes. The model building part is done using the Pandas library for datasets held in memory, or Koalas for big data. On the model serving side, the pre-processing is carried out directly with Python, using NumPy and Cython. As a result, the speed-up using both NumPy and Cython for pre-processing is around x100 compared to standard Python code. Moreover, the per-sample response time becomes similar to other compiled languages (microsecond scale).

In summary, Gators is a package to handle model building with big data ad fast real-time pre-processing, even for a large number of QPS, using only Python.

