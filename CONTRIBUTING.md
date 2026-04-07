# Contributing to Gators

Thank you for your interest in contributing to Gators! We welcome contributions from the community.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Running Tests](#running-tests)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gators.git
   cd gators
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/paypal/gators.git
   ```

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher (Python 3.14 recommended)
- pip and virtualenv

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in editable mode with dev dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify the installation**:
   ```bash
   python -c "import gators; print(gators.__version__)"
   ```

## Code Style Guidelines

We follow strict code quality standards to maintain consistency across the codebase.

### Tools We Use

- **Black** - Code formatting (line length: 100)
- **isort** - Import sorting
- **Ruff** - Fast linting
- **mypy** - Type checking (optional but encouraged)

### Before Committing

Run all code quality checks:

```bash
# Format code
black gators/
isort gators/

# Check formatting (without making changes)
black --check gators/
isort --check-only gators/

# Lint code
ruff check gators/

# Type check (optional)
mypy gators/
```

Or use tox for automated checks:

```bash
tox -e format  # Auto-format code
tox -e lint    # Run all checks
```

### Code Style Rules

- **Line length**: Maximum 100 characters
- **Import order**: Standard library → Third-party → Local (handled by isort)
- **Type hints**: Encouraged but not required
- **Docstrings**: Use Google-style docstrings for public APIs
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### Example Code

```python
import polars as pl
from typing import Optional

from gators.data_cleaning import DropColumns


class MyTransformer:
    """Brief description of the transformer.
    
    Args:
        columns: List of column names to process.
        threshold: Numeric threshold value.
    
    Examples:
        >>> transformer = MyTransformer(columns=["A", "B"])
        >>> X_transformed = transformer.fit_transform(X)
    """
    
    def __init__(self, columns: list[str], threshold: float = 0.5):
        self.columns = columns
        self.threshold = threshold
    
    def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform the dataframe."""
        # Implementation here
        return X
```

## Running Tests

### Quick Test Run

Run all tests:

```bash
pytest gators/
```

### With Coverage

```bash
pytest gators/ --cov=gators --cov-report=term-missing --cov-report=html
```

View HTML coverage report:
```bash
open htmlcov/index.html  # On macOS
# or
xdg-open htmlcov/index.html  # On Linux
```

### Test Specific Modules

```bash
# Test a specific module
pytest gators/encoders/

# Test a specific file
pytest gators/encoders/tests/test_onehot_encoder.py

# Test a specific test function
pytest gators/encoders/tests/test_onehot_encoder.py::test_basic_encoding
```

### Using Tox

Test across multiple Python versions:

```bash
# Run all environments
tox

# Run specific Python version
tox -e py314

# Run in parallel
tox -p auto
```

### Writing Tests

- Place tests in `tests/` subdirectory within each module
- Name test files: `test_*.py`
- Name test functions: `test_*`
- Use pytest fixtures for common setup
- Aim for high coverage (>85%)
- Test edge cases and error conditions

Example test structure:

```python
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import OneHotEncoder


@pytest.fixture
def sample_X():
    return pl.DataFrame({
        "category": ["A", "B", "A", "C"],
        "value": [1, 2, 3, 4]
    })


def test_basic_encoding(sample_X):
    encoder = OneHotEncoder()
    result = encoder.fit_transform(sample_X)
    
    assert "category_A" in result.columns
    assert "category_B" in result.columns
    assert result.height == 4
```

## Submitting Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer explanation if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(encoders): add target encoder with smoothing

Implements target encoding with optional smoothing parameter
to handle low-frequency categories.

Fixes #45
```

### Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Run tests and checks**:
   ```bash
   pytest gators/
   black gators/
   isort gators/
   ruff check gators/
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

6. **Open a Pull Request** on GitHub with:
   - Clear title and description
   - Link to related issues
   - Screenshots/examples if applicable
   - Test results

### PR Review Process

- All PRs require at least one approval
- CI checks must pass
- Code coverage should not decrease
- Address review comments promptly
- Squash commits before merging (if requested)

## Reporting Issues

### Bug Reports

Include:
- Gators version: `python -c "import gators; print(gators.__version__)"`
- Python version: `python --version`
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

Include:
- Clear use case description
- Proposed API/interface
- Example usage
- Why existing features don't solve the problem

### Questions

For questions about usage:
- Check existing documentation and examples
- Search closed issues
- Open a discussion (not an issue) on GitHub

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We're building this together!

## License

By contributing to Gators, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to reach out by opening a discussion on GitHub or contacting the maintainers.

Thank you for contributing to Gators! 🐊
