<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://paypal.github.io/gators/_images/GATORS_LOGO.png">
  <img alt="Gators Logo" src="https://paypal.github.io/gators/_images/GATORS_LOGO.png">
</picture>

# Gators: A Lightning-Fast Data Preprocessing And Feature Engineering Python Library


| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/gators)](https://pypi.org/project/gators/) [![Python versions](https://img.shields.io/pypi/pyversions/gators)](https://pypi.org/project/gators/) |
| Quality | [![License](https://img.shields.io/github/license/paypal/gators)](https://github.com/paypal/gators/blob/main/LICENSE) [![Coverage](https://img.shields.io/codecov/c/github/paypal/gators)](https://codecov.io/gh/paypal/gators) |
| Documentation | [![Documentation](https://img.shields.io/badge/docs-online-blue)](https://paypal.github.io/gators/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/gators)](https://pepy.tech/project/gators) [![Downloads/Month](https://static.pepy.tech/badge/gators/month)](https://pepy.tech/project/gators) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/paypal/gators?style=social)](https://github.com/paypal/gators) [![GitHub Forks](https://img.shields.io/github/forks/paypal/gators?style=social)](https://github.com/paypal/gators) [![Contributors](https://img.shields.io/github/contributors/paypal/gators)](https://github.com/paypal/gators/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/paypal/gators)](https://github.com/paypal/gators/commits/main) |


📚 **[Full Documentation](https://paypal.github.io/gators/)**


## What is Gators?

Gators is a library built on top of Polars, designed to streamline your entire ML workflow from raw data to production-ready models — benchmarked faster than scikit-learn and feature-engine across common preprocessing tasks (see Benchmarks below).

Built by the PSP Data Team at PayPal, Gators makes data preprocessing and feature engineering both **faster and simpler**.

## ⚡ Key Features

- **🚀 Lightning Fast**: Benchmarked faster than scikit-learn and feature-engine on common preprocessing tasks
- **🔄 Unified API**: Consistent sklearn-style `.fit()` and `.transform()` interface
- **📦 Production Ready**: Deploy the same Python code from notebook to production
- **🎯 Comprehensive**: 108 preprocessing transformers across 11 categories
- **🔗 Pipeline Support**: Chain transformers seamlessly with the Pipeline class
- **📤 ONNX Export**: Export fitted pipelines to ONNX for low-latency inference (most transformers supported; a handful of `feature_generation_str` transformers can't convert due to ONNX's limited string-tensor op support)
- **🎓 Easy to Learn**: If you know sklearn, you already know Gators

## 📊 Benchmarks

Gators transformers are timed head-to-head against their closest `scikit-learn`
and `feature-engine` equivalents (same algorithm, `fit` + `transform`, best-of-3
runs) on a 500,000-row synthetic dataset:

| Transformer | gators (s) | scikit-learn (s) | feature-engine (s) | speedup vs sklearn | speedup vs feature-engine |
|---|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.005 | 0.025 | 0.012 | 5.2x | 2.4x |
| StandardScaler | 0.002 | 0.009 | n/a | 3.8x | n/a |
| QuantileClipper | 0.005 | n/a | 0.056 | n/a | 12.3x |
| EqualSizeDiscretizer (5 bins) | 0.025 | 0.112 | 0.182 | 4.5x | 7.3x |
| OneHotEncoder | 0.045 | 0.283 | 0.320 | 6.2x | 7.0x |
| OrdinalEncoder | 0.029 | 0.267 | 0.127 | 9.1x | 4.3x |
| TargetEncoder | 0.029 | 0.442 | 0.163 | 15.0x | 5.5x |
| WOEEncoder | 0.028 | n/a | 0.181 | n/a | 6.5x |

`n/a` = no equivalent implementation exists in that library. Measured on an
Apple M3 Max; hardware, dataset shape, and library versions all affect
absolute numbers, so results are fully reproducible with one command:

```bash
pip install -e ".[benchmarks]"
python benchmarks/run_benchmarks.py
```

See [`benchmarks/`](benchmarks/) for full methodology, caveats, and raw results.

## �🛠️ What Can Gators Do?

### 🧹 Data Cleaning (16)
Clean and prepare your data with powerful transformers:
- `CastColumns` - Convert column data types
- `CorrelationFilter` - Remove highly correlated features
- `DropColumns` - Remove specified columns
- `DropConstantColumns` - Remove columns with constant values
- `DropDuplicateColumns` - Remove duplicate columns
- `DropDuplicateRows` - Remove duplicate rows
- `DropHighNaNRatio` - Remove columns with high missing value ratio
- `DropLowCardinality` - Remove low-cardinality columns
- `DropNearConstantColumns` - Remove near-constant columns
- `HighCardinalityFilter` - Filter high-cardinality features
- `RenameColumns` - Rename columns
- `Replace` - Replace values in data
- `RoundDigits` - Round numeric columns to a fixed number of decimal places
- `RoundSignificantDigits` - Round numeric columns to a fixed number of significant figures
- `SelectColumns` - Keep only specified columns
- `VarianceFilter` - Remove low-variance features

### ✂️ Clippers (5)
Detect and clip outliers:
- `CustomClipper` - Custom min/max bounds per column
- `GaussianClipper` - Clip based on mean ± n standard deviations
- `IQRClipper` - Clip based on interquartile range
- `MADClipper` - Clip based on median absolute deviation
- `QuantileClipper` - Clip based on quantile thresholds

### 🔢 Categorical Encoding (10)
Transform categorical variables with advanced encoding techniques:
- `BinaryEncoder` - Binary representation encoding
- `CatBoostEncoder` - CatBoost-style target encoding
- `CountEncoder` - Frequency-based encoding
- `HashEncoder` - Hashing trick for high-cardinality features
- `LeaveOneOutEncoder` - Leave-one-out target encoding
- `OneHotEncoder` - Classic one-hot encoding
- `OrdinalEncoder` - Frequency-ordered ordinal encoding
- `RareCategoryEncoder` - Replace rare/infrequent categories with a single label
- `TargetEncoder` - Target mean encoding for supervised learning
- `WOEEncoder` - Weight of Evidence encoding

### 🎯 Feature Generation - Numeric (21)
Create powerful numeric features:

**Mathematical Operations:**
- `AsymmetryIndexFeatures` - Generate asymmetry index features
- `ConcentrationIndexFeatures` - Generate concentration index features
- `DistanceFeatures` - Calculate distance features
- `EntropyFeatures` - Generate Shannon entropy features
- `FourierFeatures` - Generate Fourier basis features
- `GeneralizedRatioFeatures` - Generate generalized ratio features
- `HHIFeatures` - Herfindahl–Hirschman Index features
- `IsNull` - Generate null-indicator features
- `MathFeatures` - Apply mathematical operations between column groups
- `PlanRotationFeatures` - Rotate features in feature space
- `PolynomialFeatures` - Generate polynomial and interaction features
- `RatioFeatures` - Create ratio features between columns
- `ScalarMathFeatures` - Apply scalar operations to columns
- `WeightedSumFeatures` - Weighted sum of features

**Aggregation & Statistics:**
- `GroupLagFeatures` - Generate lag features by group
- `GroupStatisticsFeatures` - Generate group-based statistics
- `RollingStatisticsFeatures` - Generate rolling-window statistics
- `RowStatisticsFeatures` - Generate row-level statistics

**Rule-based:**
- `ComparisonFeatures` - Generate comparison features
- `ConditionFeatures` - Create conditional features
- `RuleFeatures` - Apply custom business rules


### 📝 Feature Generation - String (19)
Extract insights from text data:
- `CharacterStatistics` - Extract character-level statistics
- `CombineFeatures` - Concatenate selected string columns
- `Contains` - Binary indicator: string contains pattern
- `Endswith` - Binary indicator: string ends with pattern
- `ExtractSubstring` - Extract a fixed-position substring
- `InteractionFeatures` - Exhaustive pairwise string concatenation
- `Length` - String length
- `Lower` - Convert to lowercase
- `NGram` - Generate character or word n-gram features
- `Occurrences` - Count pattern occurrences
- `PatternDetector` - Detect regex patterns
- `RegexExtractFeatures` - Extract named groups via regex
- `Split` - Split strings on a delimiter
- `SplitExtract` - Split and extract the nth token
- `Startswith` - Binary indicator: string starts with pattern
- `StringSimilarity` - Fuzzy string similarity (Levenshtein / Jaro-Winkler)
- `TfidfFeatures` - Generate TF-IDF features
- `Upper` - Convert to uppercase
- `WordStatistics` - Extract word-level statistics

### 📅 Feature Generation - DateTime (8)
Unlock temporal patterns:
- `BusinessTimeFeatures` - Business hours/days calculations
- `CyclicFeatures` - Circular encoding for cyclical time features
- `DiffFeatures` - Calculate time differences between columns
- `DurationToDatetime` - Convert duration to datetime components
- `HolidayFeatures` - Detect and encode public holidays
- `OrdinalFeatures` - Extract year, month, day, hour, etc.
- `TimeBinFeatures` - Bin times into categorical buckets
- `TimeWindowFeatures` - Generate time-window aggregation features

### 🔄 Missing Value Imputation (6)
Handle missing data intelligently:
- `BooleanImputer` - Impute boolean columns (constant or most-frequent)
- `GroupByImputer` - Group-based imputation (median/mean per group)
- `IterativeImputer` - Multivariate iterative imputation
- `KNNImputer` - K-nearest neighbours imputation
- `NumericImputer` - Impute numeric columns (mean, median, mode, constant, forward/backward fill)
- `StringImputer` - Impute string columns (mode or constant)

### 📊 Discretization (7)
Convert continuous variables into bins:
- `CustomDiscretizer` - User-defined bin edges
- `EqualLengthDiscretizer` - Equal-width binning
- `EqualSizeDiscretizer` - Equal-frequency binning
- `GeometricDiscretizer` - Geometric progression binning
- `KMeansDiscretizer` - K-means clustering-based binning
- `QuantileDiscretizer` - Quantile-based binning
- `TreeBasedDiscretizer` - Decision tree-based optimal binning

### ⚖️ Feature Scalers (9)
Normalize and transform your features:
- `ArcSinSquareRootScaler` - Arcsine square-root transformation
- `ArcSinhScaler` - Inverse hyperbolic sine transformation
- `BoxCox` - Box-Cox power transformation
- `Log1pScaler` - Log1p scaling — log(1 + x)
- `MinmaxScaler` - Min-max normalization to [0, 1]
- `PowerScaler` - Power transformation
- `RobustScaler` - Median/IQR-based robust scaling
- `StandardScaler` - Z-score standardization
- `YeoJohnson` - Yeo-Johnson power transformation

### ✨ Feature Selection (6)
Select the most informative features:
- `CorrelationSelector` - Drop features by pairwise Pearson correlation
- `FeatureStabilitySelector` - Keep features stable across data splits
- `InformationValueSelector` - Filter by Information Value (IV)
- `MutualInformationSelector` - Filter by mutual information with target
- `PermutationImportanceSelector` - Filter by permutation feature importance
- `PSIFilter` - Filter by Population Stability Index

### 🔗 Pipeline (1)
Chain all transformers together:
- `Pipeline` - sklearn-compatible pipeline for chaining transformers

### 📤 ONNX Export
Export a fitted `Pipeline` or single transformer to a validated ONNX graph for low-latency, language-agnostic inference:

```python
from gators.onnx_converters import pipeline_to_onnx

model = pipeline_to_onnx(fitted_pipeline)  # → onnx.ModelProto
# Run with onnxruntime, Triton, or any ONNX-compatible runtime
```

Most transformers are supported, but a handful of `feature_generation_str` transformers
(`CharacterStatistics`, `NGram`, `Occurrences`, `PatternDetector`, `RegexExtractFeatures`,
`StringSimilarity`, `WordStatistics`) have no ONNX converter: they rely on variable-length
tokenization, list aggregates, or fuzzy string distance that ONNX's string-tensor op set
cannot express. Use `check_pipeline_onnx_compatibility(pipeline)` to audit a pipeline before
exporting - unsupported steps raise `OnnxNotSupportedError` (or pass through unchanged with
`errors="coerce"`).

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
    ('drop_nan',  DropHighNaNRatio(max_ratio=0.5)),  # drop columns with >50% missing values
    ('impute',    NumericImputer(strategy='median')), # fill numeric nulls with column median
    ('variance',  VarianceFilter(min_var=0.01)),      # remove near-zero-variance columns
    ('encode',    OneHotEncoder()),                   # one-hot encode all string/categorical columns
    ('scale',     StandardScaler()),                  # z-score standardize numeric columns
])

# Fit on training data, transform train + test
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed  = pipeline.transform(X_test)

# Export to ONNX for production inference
from gators.onnx_converters import pipeline_to_onnx
onnx_model = pipeline_to_onnx(pipeline)
```

## 📦 Installation

Requires Python 3.10 or higher.

```bash
pip install gators
```

With ONNX export support:

```bash
pip install "gators[onnx]"
```

Or install from source:

```bash
git clone https://github.com/paypal/gators.git
cd gators
pip install -e .
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
- **Production ML** - Export preprocessing to ONNX and run anywhere

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

