About Gators
============

What is Gators?
---------------

Gators is a high-performance machine learning preprocessing library built on top of `Polars <https://pola.rs/>`_, 
designed to streamline your entire ML workflow from raw data to production-ready models. Leveraging Polars' 
blazing-fast multi-core processing, Gators makes data preprocessing and feature engineering both **faster** and **simpler**.

Built by the PSP Data Team at PayPal, Gators solves a critical pain point: bridging the gap between Python-based 
model development and production deployment. With Gators, you can **develop and deploy using only Python** — no more 
reimplementing preprocessing logic in other languages for production!

Why Gators?
-----------

The Problem
~~~~~~~~~~~

Traditional ML workflows face a critical challenge: data preprocessing is developed in Python (pandas/sklearn) but 
often needs to be reimplemented in faster languages (C++, Java, Scala) for production deployment. This creates:

* 🔴 **Maintenance burden** - Two codebases to maintain
* 🔴 **Bugs and inconsistencies** - Different implementations can behave differently  
* 🔴 **Slower development** - Every change needs to be implemented twice

The Solution
~~~~~~~~~~~~

Gators solves this by combining:

* ✅ **Python-first development** - Write once, deploy everywhere
* ✅ **Production-grade performance** - Polars enables Rust speeds in Python
* ✅ **Unified workflow** - Same code from experimentation to production

Use Cases
---------

Gators is perfect for:

* **Fraud Detection** - Extensive feature engineering for anomaly detection
* **Risk Modeling** - Create powerful predictive features
* **Customer Analytics** - Transform complex customer data
* **Time Series** - Rich datetime feature engineering
* **NLP Tasks** - String feature extraction and encoding
* **Production ML** - Deploy preprocessing pipelines without rewriting code

Key Features
------------

* 🚀 **Lightning Fast**: Built on Polars for multi-core parallel processing
* 🔄 **Unified API**: Consistent sklearn-style ``.fit()`` and ``.transform()`` interface
* 📦 **Production Ready**: Deploy the same Python code from notebook to production
* 🎯 **Comprehensive**: 60+ preprocessing transformers covering every use case
* 🔗 **Pipeline Support**: Chain transformers seamlessly with the Pipeline class
* 🎓 **Easy to Learn**: If you know sklearn, you already know Gators

Credits
-------

Developed by the PSP Data Team at PayPal.

**Built with ⚡ by data scientists, for data scientists**
