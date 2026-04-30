# CLAUDE.md - AI Assistant Guide for Gators

**Last Updated**: April 27, 2026  
**Target**: AI coding assistants (Claude, Copilot, etc.)  
**Python Versions**: 3.10+
**Python Version to use**: 3.14
---

## Project Overview

**Gators** is a high-performance machine learning preprocessing and feature engineering library built on top of **Polars**. It provides 75+ transformers with a sklearn-compatible API for lightning-fast data transformation pipelines.

**Core Mission**: Enable ML practitioners to build fast, production-ready data pipelines with a familiar sklearn-style interface while leveraging Polars' multi-core parallel processing.

**Key Characteristics**:
- Built on Polars DataFrames (not pandas)
- Sklearn-compatible `.fit()` and `.transform()` API
- Pydantic-based configuration and validation
- Strong type hints for mypy compliance
- Performance-first design philosophy

---

## Architecture & Design Patterns

### 1. Transformer Hierarchy

All transformers inherit from `_BaseTransformer` which provides:
- Pydantic model validation via `BaseModel`
- Sklearn compatibility via `BaseEstimator` and `TransformerMixin`
- Keyword-only argument enforcement
- Custom `get_params()` and `set_params()` for Pydantic integration

```python
# Base transformer structure
class _BaseTransformer(BaseModel, BaseEstimator, TransformerMixin):
    model_config = ConfigDict(extra="forbid")
    
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        # Compute statistics, build mappings
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        # Apply transformations using Polars expressions
        return X
```

**Key Subclasses**:
- `_BaseEncoder` - For categorical encoding (adds `mapping_`, `column_mapping_`)
- Individual transformers (e.g., `NumericImputer`, `WOEEncoder`, `OneHotEncoder`)

### 2. Module Organization

```
gators/
├── transformer/          # Base classes
│   └── _base_transformer.py
├── imputers/            # Missing value handling
├── encoders/            # Categorical encoding
├── discretizers/        # Binning/bucketing
├── scalers/             # Normalization/scaling
├── feature_generation/  # Feature creation
├── feature_generation_dt/   # Datetime features
├── feature_generation_str/  # String features
├── feature_selection/   # Feature filtering
├── data_cleaning/       # Data quality
├── clippers/            # Outlier handling
└── pipeline/            # Chaining transformers
    └── pipeline.py
```

### 3. Design Principles

**Performance First**:
- Minimize calls to `X.with_columns()` - batch expressions together
- Use single-pass operations (compute all statistics at once)
- Leverage Polars' lazy evaluation and expression chaining

**Type Safety**:
- Comprehensive type hints on all methods
- Pydantic validation for configuration
- Literal types for strategy parameters

**Sklearn Compatibility**:
- All transformers implement `fit()` and `transform()`
- Compatible with sklearn pipelines (though prefer `gators.Pipeline`)
- Support `get_params()` and `set_params()`

---

## Critical Rules for AI Assistants

### ✅ DO: Performance Optimization

**ALWAYS batch Polars expressions together**:
```python
# ✅ CORRECT - Single with_columns call
transformations = [
    pl.col(col).fill_null(self._statistics[col]) 
    for col in self.subset
]
X = X.with_columns(transformations)

# ❌ WRONG - Multiple with_columns calls
for col in self.subset:
    X = X.with_columns(pl.col(col).fill_null(self._statistics[col]))
```

**Compute statistics in single pass**:
```python
# ✅ CORRECT - All medians at once
median_results = X.select([pl.col(c).median() for c in self.subset]).row(0)
self._statistics = {col: median_results[i] for i, col in enumerate(self.subset)}

# ❌ WRONG - Loop through columns
for col in self.subset:
    self._statistics[col] = X[col].median()
```

### ✅ DO: Follow Naming Conventions

- **Class names**: PascalCase (e.g., `NumericImputer`, `WOEEncoder`)
- **Methods**: snake_case (e.g., `fit`, `transform`, `get_params`)
- **Private attributes**: Prefix with `_` and use `PrivateAttr()` from Pydantic
- **Generated columns**: Use descriptive suffixes (e.g., `{col}__impute_{strategy}`)

### ✅ DO: Preserve Type Hints & Pydantic Validation

```python
from typing import Dict, List, Literal, Optional, Union
from pydantic import PrivateAttr, Field

class MyTransformer(_BaseTransformer):
    # Public parameters - Pydantic fields
    strategy: Literal['mean', 'median', 'mode']
    subset: Optional[List[str]] = None
    inplace: bool = True
    
    # Private attributes - not in __init__
    _statistics: Dict[str, float] = PrivateAttr(default_factory=dict)
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)
```

### ✅ DO: Maintain Sklearn API Compatibility

```python
def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "TransformerName":
    """Always return self from fit()"""
    # Compute statistics here
    return self

def transform(self, X: pl.DataFrame) -> pl.DataFrame:
    """Never modify X in-place unless inplace=True parameter"""
    # Apply transformations
    return X
```

### ✅ DO: Write Comprehensive Docstrings

Follow NumPy/Scipy style with:
- One-line summary
- Detailed parameter descriptions with types
- Returns section
- Examples section with executable code
- Notes/Warnings if applicable

```python
def transform(self, X: pl.DataFrame) -> pl.DataFrame:
    """Transform the input DataFrame by imputing missing values.

    Parameters
    ----------
    X : pl.DataFrame
        Input DataFrame with numeric columns containing null values.

    Returns
    -------
    pl.DataFrame
        DataFrame with imputed numeric columns.
    """
```

### ❌ DON'T: Common Pitfalls

1. **Don't use positional arguments** - `_BaseTransformer` enforces keyword-only
2. **Don't modify input DataFrames** - unless `inplace=True` is set
3. **Don't use pandas operations** - this is a Polars-based library
4. **Don't forget type hints** - mypy compliance is required
5. **Don't skip validation** - leverage Pydantic's validation features

---

## Key Technologies & Patterns

### Polars DataFrame Operations

**Expression-based transformations**:
```python
# Building multiple transformations
transformations = [
    pl.col(col).fill_null(strategy='mean') for col in numeric_cols
]
X = X.with_columns(transformations)

# Conditional logic
pl.when(condition).then(value).otherwise(default)

# Aggregations
X.select([pl.col(c).median() for c in cols])
```

**Important Polars methods used**:
- `.with_columns()` - Add/modify columns (minimize calls!)
- `.select()` - Select/compute columns
- `.drop()` - Remove columns
- `.fill_null()` - Impute missing values
- `.replace_strict()` - Replace values with mapping
- `.cast()` - Type conversion
- `.alias()` - Rename expressions

### Pydantic Models

**Configuration**:
```python
model_config = ConfigDict(
    extra="forbid",  # No unexpected parameters
    arbitrary_types_allowed=True  # For Pipeline steps
)
```

**Private attributes** (not constructor parameters):
```python
_statistics: Dict[str, float] = PrivateAttr(default_factory=dict)
```

**Field validation**:
```python
from pydantic import Field, PositiveInt, PositiveFloat

min_count: Union[PositiveInt, PositiveFloat] = 1
mapping_: Dict[str, Dict[str, float]] = Field(default_factory=dict)
```

### Testing with Pytest

**Standard test structure**:
```python
import polars as pl
import pytest
from polars.testing import assert_frame_equal

@pytest.fixture
def sample_dataframe():
    return pl.DataFrame({...})

def test_transformer_basic(sample_dataframe):
    transformer = MyTransformer(param=value)
    transformer.fit(sample_dataframe)
    result = transformer.transform(sample_dataframe)
    expected = pl.DataFrame({...})
    assert_frame_equal(result, expected)
```

**Test coverage expectations**:
- Basic functionality
- Edge cases (empty DataFrames, all nulls, etc.)
- Parameter combinations (`inplace=True/False`, `drop_columns=True/False`)
- Subset selection
- Sklearn compatibility (`get_params`, `set_params`)

---

## Common Tasks & Workflows

### Adding a New Transformer

1. **Create the class**:
   ```python
   from ..transformer._base_transformer import _BaseTransformer
   
   class MyTransformer(_BaseTransformer):
       # Define parameters as Pydantic fields
       strategy: Literal['option1', 'option2']
       subset: Optional[List[str]] = None
       
       # Private attributes
       _stats: Dict = PrivateAttr(default_factory=dict)
   ```

2. **Implement `fit()` method**:
   - Compute statistics in single pass
   - Store in private attributes
   - Return `self`

3. **Implement `transform()` method**:
   - Build list of Polars expressions
   - Apply with single `X.with_columns(transformations)`
   - Handle `inplace` and `drop_columns` parameters

4. **Add comprehensive docstring** with examples

5. **Write tests** covering all parameter combinations

6. **Add to `__init__.py`** for public API

### Optimizing Polars Expressions

**Before optimization**:
```python
for col in columns:
    X = X.with_columns(pl.col(col).some_operation())
```

**After optimization**:
```python
transformations = [pl.col(col).some_operation() for col in columns]
X = X.with_columns(transformations)
```

### Debugging Type Hint Issues

- Use `# type: ignore[specific-error]` sparingly and with comments
- Check Pydantic field types match usage
- Verify return types match method signatures
- Run `mypy gators/` to catch issues early

### Adding New Encoding Strategies

1. Inherit from `_BaseEncoder` (not `_BaseTransformer`)
2. Implement `fit()` to populate `self.mapping_` dictionary
3. Use inherited `transform()` method (handles mapping application)
4. Handle boolean columns (cast to string first)
5. Support `min_count` filtering if applicable

---

## Project-Specific Gotchas

### 1. Python 3.14 Compatibility
- Ensure all code works with Python 3.14 features
- Check compatibility of dependencies (Polars, Pydantic, etc.)
- Use modern type hints (e.g., `list[str]` instead of `List[str]` where appropriate)

### 2. Minimal `X.with_columns()` Calls
**Critical performance consideration**: Each `X.with_columns()` call creates a new DataFrame. Always batch expressions:
```python
# Single call with all transformations
X = X.with_columns([expr1, expr2, expr3, ...])
```

### 3. Boolean Column Handling
When using `.replace_strict()` with boolean columns, cast to string first:
```python
if X[col].dtype == pl.Boolean:
    string_mapping = {str(k).lower(): v for k, v in mapping.items()}
    expr = pl.col(col).cast(pl.String).replace_strict(string_mapping, ...)
```

### 4. Pydantic Private Attributes
Private attributes must use `PrivateAttr()` and won't appear in `__init__`:
```python
_statistics: Dict[str, float] = PrivateAttr(default_factory=dict)
```

### 5. Pipeline vs sklearn Pipeline
Prefer `gators.Pipeline` over `sklearn.pipeline.Pipeline` for Polars DataFrames - it's optimized to avoid unnecessary type conversions.

### 6. Subset Auto-Detection
When `subset=None`, auto-detect applicable columns in `fit()`:
```python
if not self.subset:
    self.subset = [
        col for col, dtype in zip(X.columns, X.dtypes)
        if dtype not in [pl.String, pl.Boolean]
    ]
```

---

## Testing Strategy

### Test File Organization
```
gators/
└── module_name/
    ├── __init__.py
    ├── transformer1.py
    ├── transformer2.py
    └── tests/
        ├── test_transformer1.py
        └── test_transformer2.py
```

### Running Tests
```bash
# Run all tests
python3.14 -m pytest

# Run with coverage
python3.14 -m pytest --cov=gators --cov-report=html

# Run specific test file
python3.14 -m pytest gators/imputers/tests/test_numeric_imputer.py

# Run specific test
python3.14 -m pytest gators/imputers/tests/test_numeric_imputer.py::test_imputer_constant
```

### Test Guidelines
- Use `polars.testing.assert_frame_equal()` for DataFrame comparisons
- Test both `inplace=True` and `inplace=False` modes
- Test `drop_columns` parameter variations
- Test with `subset` parameter (specific columns)
- Test edge cases: empty DataFrames, all nulls, single column
- Verify sklearn compatibility: `get_params()`, `set_params()`, pipeline integration

---

## Development Setup

### Installation
```bash
# Clone repository
git clone https://github.com/paypal/gators.git
cd gators

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies
- **Core**: polars, pydantic, pyarrow, scikit-learn
- **Dev**: pytest, pytest-cov, black, isort
- **Optional**: jupyter, matplotlib (for notebooks)

### Code Style
- **Formatter**: black (line length 100)
- **Import sorting**: isort
- **Type checking**: mypy (when enabled)
- **Docstrings**: NumPy/Scipy style

### Pre-commit Checks
```bash
# Format code
black gators/

# Sort imports
isort gators/

# Run tests
pytest

# Generate coverage
pytest --cov=gators --cov-report=html
```

---

## Code Examples

### Example 1: Basic Transformer Usage
```python
import polars as pl
from gators.imputers import NumericImputer

# Create DataFrame
X = pl.DataFrame({
    'A': [1.0, 2.0, None, 4.0],
    'B': [5.0, None, 7.0, 8.0]
})

# Fit and transform
imputer = NumericImputer(strategy='mean', inplace=False)
imputer.fit(X)
X_transformed = imputer.transform(X)
```

### Example 2: Pipeline Usage
```python
from gators.pipeline import Pipeline
from gators.imputers import NumericImputer, StringImputer
from gators.encoders import WOEEncoder

steps = [
    ('num_impute', NumericImputer(strategy='median')),
    ('str_impute', StringImputer(strategy='constant', value='MISSING')),
    ('woe_encode', WOEEncoder(subset=['category_col']))
]

pipe = Pipeline(steps=steps)
pipe.fit(X_train, y=y_train)
X_transformed = pipe.transform(X_test)
```

### Example 3: Custom Transformer Template
```python
from typing import Dict, List, Optional
import polars as pl
from pydantic import PrivateAttr
from ..transformer._base_transformer import _BaseTransformer

class CustomTransformer(_BaseTransformer):
    """One-line description.
    
    Parameters
    ----------
    param1 : type
        Description.
    subset : Optional[List[str]], default=None
        Columns to transform.
    inplace : bool, default=True
        Whether to modify columns in-place.
    """
    
    param1: str
    subset: Optional[List[str]] = None
    inplace: bool = True
    _computed_stats: Dict = PrivateAttr(default_factory=dict)
    
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CustomTransformer":
        if not self.subset:
            self.subset = X.columns
        
        # Compute statistics in single pass
        stats = X.select([
            pl.col(c).some_operation() for c in self.subset
        ]).row(0)
        
        self._computed_stats = {
            col: stats[i] for i, col in enumerate(self.subset)
        }
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        # Build all transformations
        transformations = [
            pl.col(col).apply_operation(self._computed_stats[col])
            for col in self.subset
        ]
        
        # Single with_columns call
        return X.with_columns(transformations)
```

---

## Questions & Answers

**Q: When should I use `_BaseEncoder` vs `_BaseTransformer`?**  
A: Use `_BaseEncoder` for categorical encoding tasks that need `mapping_` and `column_mapping_` attributes. Use `_BaseTransformer` for all other transformations.

**Q: How do I handle parameters that shouldn't be in `__init__`?**  
A: Use Pydantic's `PrivateAttr()` for computed/fitted attributes that are set during `fit()`.

**Q: Should I modify the input DataFrame `X`?**  
A: No, unless `inplace=True`. Always create transformations and apply them, returning the result.

**Q: How do I ensure performance?**  
A: Batch all Polars expressions into a single list and call `X.with_columns()` once. Compute all statistics in single-pass operations.

**Q: What if a transformer doesn't need `fit()`?**  
A: Still implement `fit()` that just returns `self` for sklearn compatibility.

---

## Summary Checklist

When working on Gators code, ensure:

- [ ] Inherits from `_BaseTransformer` or `subfolder/_Base...`
- [ ] All parameters are Pydantic fields with type hints
- [ ] Private attributes use `PrivateAttr()`
- [ ] `fit()` returns `self`
- [ ] Transformations batched into single `X.with_columns()` call
- [ ] Comprehensive docstring with examples
- [ ] Tests cover all parameter combinations
- [ ] Compatible with sklearn API (`get_params`, `set_params`)
- [ ] Code formatted with black (line length 100)
- [ ] Imports sorted with isort
- [ ] Works with Python 3.10+
- [ ] Type hints for mypy compliance

---

**End of CLAUDE.md** - For questions or updates, see [CONTRIBUTING.md](CONTRIBUTING.md)
