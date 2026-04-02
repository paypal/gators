# 🐊 Gators

[![PyPI version](https://img.shields.io/pypi/v/gators)](https://pypi.org/project/gators/)
[![Python versions](https://img.shields.io/pypi/pyversions/gators)](https://pypi.org/project/gators/)
[![License](https://img.shields.io/github/license/paypal/gators)](https://github.com/paypal/gators/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/paypal/gators)](https://codecov.io/gh/paypal/gators)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

> **Lightning-fast data preprocessing and feature engineering for machine learning**

## What is Gators?

Gators is a high-performance machine learning preprocessing library built on top of [Polars](https://pola.rs/), designed to streamline your entire ML workflow from raw data to production-ready models. Leveraging Polars' blazing-fast multi-core processing, Gators makes data preprocessing and feature engineering both **faster** and **simpler**.

Built by the PSP Data Team at PayPal, Gators solves a critical pain point: bridging the gap between Python-based model development and production deployment. With Gators, you can **develop and deploy using only Python** — no more reimplementing preprocessing logic in other languages for production!

## ⚡ Key Features

- **🚀 Lightning Fast**: Built on Polars for multi-core parallel processing
- **🔄 Unified API**: Consistent sklearn-style `.fit()` and `.transform()` interface
- **🎯 Comprehensive**: 60+ preprocessing transformers covering every use case
- **🔗 Pipeline Support**: Chain transformers seamlessly with the Pipeline class
- **🎓 Easy to Learn**: If you know sklearn, you already know Gators

## 🛠️ What Can Gators Do?

### 🧹 Data Cleaning
Clean and prepare your data with powerful transformers.

### 🔢 Categorical Encoding
Transform categorical variables with advanced encoding techniques.

### 🎯 Feature Generation - Numeric
Create powerful numeric features.

### 📝 Feature Generation - String
Extract insights from text data.

### 📅 Feature Generation - DateTime
Unlock temporal patterns.

### 🔄 Missing Value Imputation
Handle missing data intelligently.

### 📊 Discretization
Convert continuous variables into bins.

### ⚖️ Feature Scaling
Normalize your features.

### 🔗 Pipeline
Chain all transformers together:

## 🚀 Quick Start

```python
import polars as pl
from gators.data_cleaning import DropHighNaNRatio, VarianceFilter
from gators.encoders import OneHotEncoder
from gators.imputers import NumericImputer
from gators.scalers import StandardScaler
from gators.pipeline import Pipeline

# Load your data
X = pl.read_csv("data.csv")

# Build a preprocessing pipeline
pipeline = Pipeline([
    ('drop_nan', DropHighNaNRatio(threshold=0.5)),
    ('impute', NumericImputer(strategy='median')),
    ('variance', VarianceFilter(threshold=0.01)),
    ('encode', OneHotEncoder()),
    ('scale', StandardScaler())
])

# Fit and transform
X_processed = pipeline.fit_transform(X)

# Deploy the same pipeline in production!
```

## 📦 Installation

```bash
pip install gators
```

Or install from source:

```bash
git clone https://github.com/paypal/gators.git
cd gators
pip install -e .
```

## 🎯 Use Cases

Gators is perfect for:

- **Data Preprocessing** - End-to-end data cleaning and feature engineering
- **Risk Modeling** - Create powerful predictive features
- **Customer Analytics** - Transform complex customer data
- **Time Series** - Rich datetime feature engineering
- **NLP Tasks** - String feature extraction and encoding

## 🏗️ Why Gators?

Gators is a **lightning-fast data preprocessing and feature engineering** library built on top of  [Polars](https://pola.rs/), 
designed to streamline your entire ML workflow from raw data to production-ready models. Leveraging Polars' 
blazing-fast multi-core processing.

Built by the PSP Data Team at PayPal, Gators makes data preprocessing and feature engineering both **faster** and **simpler**.


## 🤝 Contributing

We welcome contributions! Please check out our contributing guidelines.

## 📄 License

Gators is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## 🙏 Credits

Developed by the PSP Data Team at PayPal.

---

**Built by data scientists, for data scientists**
