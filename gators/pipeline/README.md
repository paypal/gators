# Gators Pipeline

A lightweight Pipeline implementation specifically designed for Gators transformers that work with Polars DataFrames.

## Overview

The `gators.pipeline.Pipeline` class provides a sklearn-compatible API for chaining Gators transformers without the type conversion and validation overhead of sklearn's Pipeline. This is particularly important when working with Polars DataFrames that contain mixed data types (strings, numerics, etc.).

## Key Features

- **Polars-Native**: Designed specifically for Polars DataFrames
- **sklearn-Compatible API**: Familiar `fit()`, `transform()`, and `fit_transform()` methods
- **No Type Conversion**: Passes DataFrames between steps without validation overhead
- **Verbose Mode**: Optional progress printing for debugging
- **Full sklearn BaseEstimator**: Supports `get_params()` and `set_params()` for hyperparameter tuning

## Usage

### Basic Example

```python
from gators.pipeline import Pipeline
from gators.imputers import NumericImputer, StringImputer
from gators.encoders import WOEEncoder
import polars as pl

# Define your pipeline steps
steps = [
    ('numeric_imputer', NumericImputer(strategy='median', inplace=True)),
    ('string_imputer', StringImputer(strategy='constant', value='MISSING', inplace=True)),
    ('woe_encoder', WOEEncoder(columns=['cat_col1', 'cat_col2'], inplace=True))
]

# Create the pipeline
pipe = Pipeline(steps=steps, verbose=True)

# Fit and transform
pipe.fit(X_train, y=y_train)
X_transformed = pipe.transform(X_train)

# Or use fit_transform
X_transformed = pipe.fit_transform(X_train, y=y_train)
```

### Complete Preprocessing Pipeline

```python
from gators.pipeline import Pipeline
from gators.data_cleaning import (
    Replace,
    DropHighNaNRatio,
    DropLowCardinality,
    VarianceFilter,
    CastColumns
)
from gators.feature_generation_str import Lower
from gators.imputers import NumericImputer, StringImputer
from gators.encoders import WOEEncoder, CountEncoder, OneHotEncoder

steps = [
    ('cast', CastColumns(columns=['bool_col'], dtype=pl.String, inplace=True)),
    ('drop_nan', DropHighNaNRatio(max_ratio=0.9)),
    ('drop_low_card', DropLowCardinality(min_count=2)),
    ('variance', VarianceFilter(min_var=0.0001)),
    ('replace', Replace(to_replace={' ': '-'}, inplace=True)),
    ('lower', Lower(inplace=True)),
    ('num_impute', NumericImputer(strategy='median', inplace=True)),
    ('str_impute', StringImputer(strategy='constant', value='MISSING', inplace=True)),
    ('onehot', OneHotEncoder(columns=['cat1', 'cat2'], drop_columns=True)),
    ('count', CountEncoder(columns=['cat3', 'cat4'], drop_columns=False)),
    ('woe', WOEEncoder(columns=['cat3', 'cat4'], inplace=True)),
]

pipe = Pipeline(steps=steps, verbose=True)
pipe.fit(X_train, y=y_train)
X_train_transformed = pipe.transform(X_train)
X_test_transformed = pipe.transform(X_test)
```

### Accessing Pipeline Components

```python
# Access by name
imputer = pipe.named_steps['numeric_imputer']

# Access by index
first_step = pipe[0]

# Get a sub-pipeline
sub_pipe = pipe[0:3]

# Iterate over steps
for name, transformer in pipe.steps:
    print(f"{name}: {transformer}")
```

### Parameter Management

```python
# Get all parameters
params = pipe.get_params(deep=True)

# Set parameters
pipe.set_params(
    numeric_imputer__strategy='mean',
    verbose=True
)
```

## Differences from sklearn.pipeline.Pipeline

1. **No Type Conversion**: Gators Pipeline doesn't perform intermediate type checks or conversions, allowing Polars DataFrames with mixed types to flow through the pipeline without issues.

2. **Polars-First**: Optimized for Polars DataFrame operations and Gators transformers.

3. **Simplified**: Focuses on the core fit/transform workflow without memory management or caching features.

4. **Always Chains**: All transformers are always applied (no 'passthrough' option).

## When to Use

Use `gators.pipeline.Pipeline` when:
- Working with Gators transformers and Polars DataFrames
- You encounter type conversion errors with sklearn's Pipeline
- You want a lightweight, Polars-native alternative

Use `sklearn.pipeline.Pipeline` when:
- Working with sklearn transformers and numpy/pandas
- You need advanced features like caching or memory management
- You're working in a pure sklearn ecosystem

## API Reference

### Class: Pipeline

```python
class Pipeline(steps, verbose=False)
```

**Parameters:**
- `steps` (list of tuples): List of (name, transformer) tuples to chain
- `verbose` (bool, default=False): If True, prints step names during execution

**Attributes:**
- `steps`: The list of (name, transformer) tuples
- `named_steps`: Dictionary mapping step names to transformers

**Methods:**
- `fit(X, y=None)`: Fit all transformers
- `transform(X)`: Transform data through all transformers
- `fit_transform(X, y=None)`: Fit and transform in one call
- `get_params(deep=True)`: Get parameters
- `set_params(**params)`: Set parameters

## Examples

See the examples directory for complete working examples with real datasets.
