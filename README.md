<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://paypal.github.io/gators/_images/GATORS_LOGO.png">
  <img alt="Gators Logo" src="https://paypal.github.io/gators/_images/GATORS_LOGO.png">
</picture>

# Gators: A Lightning-Fast Data Preprocessing And Feature Engineering Python Library


[![PyPI version](https://img.shields.io/pypi/v/gators)](https://pypi.org/project/gators/)
[![Python versions](https://img.shields.io/pypi/pyversions/gators)](https://pypi.org/project/gators/)
[![License](https://img.shields.io/github/license/paypal/gators)](https://github.com/paypal/gators/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/paypal/gators)](https://codecov.io/gh/paypal/gators)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://paypal.github.io/gators/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Downloads](https://static.pepy.tech/badge/gators)](https://pepy.tech/project/gators)
[![Downloads/Month](https://static.pepy.tech/badge/gators/month)](https://pepy.tech/project/gators)
[![GitHub Stars](https://img.shields.io/github/stars/paypal/gators?style=social)](https://github.com/paypal/gators)
[![GitHub Forks](https://img.shields.io/github/forks/paypal/gators?style=social)](https://github.com/paypal/gators)
[![Contributors](https://img.shields.io/github/contributors/paypal/gators)](https://github.com/paypal/gators/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/paypal/gators)](https://github.com/paypal/gators/commits/main)

|

📚 **[Full Documentation](https://paypal.github.io/gators/)**


## What is Gators?

Gators is a library built on top of Polars, designed to streamline your entire ML workflow from raw data to production-ready models, leveraging **Polars' blazing-fast multi-core processing**.

Built by the PSP Data Team at PayPal, Gators makes data preprocessing and feature engineering both **faster and simpler**.

## ⚡ Key Features

- **🚀 Lightning Fast**: Built on Polars for multi-core parallel processing
- **🔄 Unified API**: Consistent sklearn-style `.fit()` and `.transform()` interface
- **📦 Production Ready**: Deploy the same Python code from notebook to production
- **🎯 Comprehensive**: 75+ preprocessing transformers covering every use case
- **🔗 Pipeline Support**: Chain transformers seamlessly with the Pipeline class
- **🎓 Easy to Learn**: If you know sklearn, you already know Gators

## 🛠️ What Can Gators Do?

### 🧹 Data Cleaning
Clean and prepare your data with powerful transformers:
- `CastColumns` - Convert column data types
- `CorrelationFilter` - Remove highly correlated features
- `DropColumns` - Remove specified columns
- `DropConstantColumns` - Remove columns with constant values
- `DropDuplicateColumns` - Remove duplicate columns
- `DropDuplicateRows` - Remove duplicate rows
- `DropHighNaNRatio` - Remove columns with high missing value ratio
- `DropLowCardinality` - Remove low cardinality columns
- `HighCardinalityFilter` - Filter high cardinality features
- `OutlierFilter` - Detect and filter outliers
- `RenameColumns` - Rename columns
- `Replace` - Replace values in data
- `VarianceFilter` - Remove low variance features

### 🔢 Categorical Encoding
Transform categorical variables with advanced encoding techniques:
- `BinaryEncoder` - Binary representation encoding
- `CatBoostEncoder` - CatBoost-style encoding
- `CountEncoder` - Frequency-based encoding
- `LeaveOneOutEncoder` - Leave-one-out encoding
- `OneHotEncoder` - Classic one-hot encoding
- `OrdinalEncoder` - Order-based encoding
- `RareCategoryEncoder` -  Replace rare/infrequent categories by a single category
- `TargetEncoder` - Target-based encoding for supervised learning
- `WOEEncoder` - Weight of Evidence encoding

### 🎯 Feature Generation - Numeric
Create powerful numeric features:
**Mathematical Operations:**
- `DistanceFeatures` - Calculate distance features
- `IsNull` - Generate null indicator features
- `MathFeatures` - Apply mathematical operations (add, subtract, multiply, divide)
- `RatioFeatures` - Create ratio features between columns
- `PlaneRotationFeatures` - Rotate features in feature space
- `PolynomialFeatures` - Generate polynomial combinations
- `ScalarMathFeatures` - Apply scalar operations

**Aggregation & Statistics:**
- `GroupLagFeatures` - Generate lag features by group
- `GroupScalingFeatures` - Scale features within groups
- `GroupStatisticsFeatures` - Calculate group statistics
- `RowStatisticsFeatures` - Calculate row-wise statistics

**Rule-based**  
- `ComparisonFeatures` - Generate comparison features
- `ConditionFeatures` - Create conditional features
- `RuleFeatures` - Apply custom business rules



### 📝 Feature Generation - String
Extract insights from text data:
- `CharacterStatistics` - Extract character-level statistics
- `CombineFeatures` - Combine string features
- `Contains` - Check if string contains pattern
- `Endswith` - Check if string ends with pattern
- `ExtractSubstring` - Extract substring from text
- `InteractionFeatures` - Generate string interaction features
- `Length` - Calculate string length
- `Lower` - Convert text to lowercase
- `NGram` - Generate n-gram features
- `Occurrences` - Count pattern occurrences
- `PatternDetector` - Detect patterns in text
- `Split` - Split strings
- `SplitExtract` - Split and extract from strings
- `Startswith` - Check if string starts with pattern
- `Upper` - Convert text to uppercase

### 📅 Feature Generation - DateTime
Unlock temporal patterns:
- `BusinessTimeFeatures` - Business hours/days calculations
- `CyclicFeatures` - Circular encoding for cyclical time features
- `DiffFeatures` - Calculate time differences
- `DurationToDatetime` - Convert duration to datetime
- `HolidayFeatures` - Detect and encode holidays
- `OrdinalFeatures` - Extract year, month, day, hour, etc.
- `TimeBinFeatures` - Bin times into categories
- `TimeWindowFeatures` - Generate time window features

### 🔄 Missing Value Imputation
Handle missing data intelligently:
- `BooleanImputer` - Impute boolean columns
- `GroupByImputer` - Group-based imputation strategies
- `NumericImputer` - Impute numeric columns (mean, median, mode, constant)
- `StringImputer` - Impute string columns (mode, constant)

### 📊 Discretization
Convert continuous variables into bins:
- `CustomDiscretizer` - Custom bin edges
- `EqualLengthDiscretizer` - Equal-width binning
- `EqualSizeDiscretizer` - Equal-frequency binning
- `GeometricDiscretizer` - Geometric progression binning
- `KMeansDiscretizer` - K-means clustering-based binning
- `QuantileDiscretizer` - Quantile-based binning
- `TreeBasedDiscretizer` - Decision tree-based binning

### ⚖️ Feature Scaling
Normalize your features:
- `ArcsinSquarerootScaler` - Arcsine square root transformation
- `ArcsinhScaler` - Inverse hyperbolic sine transformation
- `BoxCox` - Box-Cox power transformation
- `LogScaler` - Logarithmic scaling
- `MinmaxScaler` - Min-max normalization
- `PowerScaler` - Power transformation
- `StandardScaler` - Standardization (z-score normalization)
- `YeoJohnson` - Yeo-Johnson power transformation

### 🔗 Pipeline
Chain all transformers together:
- `Pipeline` - sklearn-compatible pipeline for chaining transformers

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
pipeline = Pipeline(steps=[
    ('drop_nan', DropHighNaNRatio(max_ratio=0.5)),
    ('impute', NumericImputer(strategy='median')),
    ('variance', VarianceFilter(min_var=0.01)),
    ('encode', OneHotEncoder()),  # One-hot encode ALL the String or Categorical columns    
    ('scale', StandardScaler())
])

# Fit and transform
X_processed = pipeline.fit_transform(X)

# Serialize the pipeline with pickle/joblib for production deployment
```

## 📦 Installation

Requires Python 3.10 or higher.

```bash
pip3 install gators
```

Or install from source:

```bash
git clone https://github.com/paypal/gators.git
cd gators
pip3 install -e .    # Install in editable/development mode
```

## 📚 Documentation

For detailed documentation, tutorials, and API reference, visit:

**[https://paypal.github.io/gators/](https://paypal.github.io/gators/)**

## 🎯 Use Cases

Gators is perfect for:

- **Fraud Detection** - Extensive feature engineering for anomaly detection
- **Risk Modeling** - Create powerful predictive features
- **Customer Analytics** - Transform complex customer data
- **Time Series** - Rich datetime feature engineering
- **NLP Tasks** - String feature extraction and encoding

## 🏢 Used By

Gators powers ML pipelines at:
- PayPal (internal use)

## 🤝 Contributing

We welcome contributions! Please check out our [contributing guidelines](https://github.com/paypal/gators/blob/master/CONTRIBUTING.md).

## 📄 License

Gators is licensed under the Apache License 2.0. See [LICENSE](https://github.com/paypal/gators/blob/master/LICENSE) file for details.

## 🙏 Credits

Developed by the PSP Data Team at PayPal.

---

**Built by data scientists, for data scientists**