# CLAUDE.md - AI Assistant Guide for Gators

**Last Updated**: April 27, 2026
**Target**: AI coding assistants (Claude, Copilot, etc.)
**Python Versions**: 3.10+ (use 3.14 for development)

---

## Project Overview

**Gators** is a high-performance ML preprocessing and feature-engineering library built on
**Polars**, providing 75+ transformers with a sklearn-compatible (`.fit()`/`.transform()`) API.

- Built on Polars DataFrames (not pandas)
- Pydantic-based configuration and validation
- Strong type hints for mypy compliance
- Performance-first: minimize DataFrame copies, batch expressions, single-pass statistics

---

## Architecture

All transformers inherit from `_BaseTransformer(BaseModel, BaseEstimator, TransformerMixin)`:

```python
class _BaseTransformer(BaseModel, BaseEstimator, TransformerMixin):
    model_config = ConfigDict(extra="forbid")  # keyword-only args enforced

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None):
        # Compute statistics, build mappings, populate private attrs
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        # Apply transformations using batched Polars expressions
        return X
```

- `_BaseEncoder` — categorical encoding subclass (adds `mapping_`, `_column_mapping`).
- Module layout: `transformer/` (base classes), `imputers/`, `encoders/`, `discretizers/`,
  `scalers/`, `feature_generation/` (+ `_dt`/`_str` variants), `feature_selection/`,
  `data_cleaning/`, `clippers/`, `pipeline/`, `onnx_converters/`.

---

## Core Rules

1. **Batch `with_columns()` calls** — one call per `fit()`/`transform()`, never loop:
   ```python
   # CORRECT
   X = X.with_columns([pl.col(c).fill_null(self._statistics[c]) for c in self.subset])
   # WRONG
   for c in self.subset:
       X = X.with_columns(pl.col(c).fill_null(self._statistics[c]))
   ```
2. **Compute statistics in a single pass** — e.g. `X.select([pl.col(c).median() for c in cols]).row(0)`,
   not a per-column loop.
3. **Keyword-only, Pydantic-typed parameters** — no positional args; public params are typed
   Pydantic fields (`Literal` for strategy enums); fitted/computed state uses
   `PrivateAttr(default_factory=...)` (never appears in `__init__`).
4. **Never mutate `X` in place** unless the transformer has an explicit `inplace=True` param.
5. **`fit()` always returns `self`** (even if it's a no-op, for sklearn compatibility).
6. **No pandas** — this is a Polars-only library.
7. **Naming**: classes `PascalCase`, methods `snake_case`, private attrs `_prefixed` +
   `PrivateAttr()`, generated columns use descriptive suffixes (e.g. `{col}__impute_{strategy}`).
8. **Subset auto-detection**: when `subset=None`, detect applicable columns in `fit()`:
   ```python
   if not self.subset:
       self.subset = [c for c, dt in zip(X.columns, X.dtypes) if dt not in (pl.String, pl.Boolean)]
   ```
9. **Boolean columns with `.replace_strict()`**: cast to string first (title/lower-case keys
   must match, e.g. `{str(k).lower(): v for k, v in mapping.items()}`).
10. **Prefer `gators.Pipeline`** over `sklearn.pipeline.Pipeline` — avoids unneeded conversions.
11. **NumPy/Scipy-style docstrings**: one-line summary, `Parameters`/`Returns`, an executable
    `Examples` block.

---

## Adding a New Transformer

1. Inherit from `_BaseTransformer` (or `_BaseEncoder` for categorical encoding).
2. Declare public params as typed Pydantic fields; private/fitted state as `PrivateAttr()`.
3. `fit()`: compute all statistics in one pass, store in private attrs, `return self`.
4. `transform()`: build a list of expressions, apply via a single `X.with_columns(...)`;
   respect `inplace`/`drop_columns` params if present.
5. Add a NumPy-style docstring with examples; write tests (see below); export from `__init__.py`.

Template:

```python
class CustomTransformer(_BaseTransformer):
    """One-line description.

    Parameters
    ----------
    param1 : type
        Description.
    subset : list[str], default=None
        Columns to transform.
    inplace : bool, default=True
        Whether to modify columns in-place.
    """

    param1: str
    subset: list[str] | None = None
    inplace: bool = True
    _computed_stats: dict = PrivateAttr(default_factory=dict)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "CustomTransformer":
        if not self.subset:
            self.subset = X.columns
        stats = X.select([pl.col(c).some_operation() for c in self.subset]).row(0)
        self._computed_stats = {col: stats[i] for i, col in enumerate(self.subset)}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        transformations = [
            pl.col(c).apply_operation(self._computed_stats[c]) for c in self.subset
        ]
        return X.with_columns(transformations)
```

---

## Testing

- Layout: `gators/<module>/tests/test_<transformer>.py`, using `pytest` +
  `polars.testing.assert_frame_equal`.
- Cover: basic functionality, edge cases (empty DataFrame, all-nulls), `inplace`/`drop_columns`
  combinations, `subset` selection, and sklearn compatibility (`get_params`/`set_params`).

```bash
python3.14 -m pytest                                  # all tests
python3.14 -m pytest --cov=gators --cov-report=html   # with coverage
python3.14 -m pytest gators/imputers/tests/test_numeric_imputer.py::test_imputer_constant
```

---

## Development Setup

```bash
git clone https://github.com/paypal/gators.git && cd gators
pip install -e ".[dev]"
```

- **Core deps**: polars, pydantic, pyarrow, scikit-learn.
- **Style**: Ruff for lint + format (line length 100, replaces black/isort); mypy for type
  checking. A `.pre-commit-config.yaml` runs these automatically — `pre-commit install` once.
- Manual check: `ruff check gators/ && ruff format gators/ && mypy gators/ && pytest`.

---

## Gotchas

- **Pydantic private attrs**: must use `PrivateAttr()` — a bare `_attr: type = default` on a
  `BaseModel` subclass will not behave as a normal instance attribute.
- **`X.with_columns()` cost**: each call materializes a new DataFrame — always batch.
- **Python 3.14**: target version for this repo; use modern generics (`list[str]`, `X | None`).

---

## Token-Efficiency Guidelines for AI Assistants

- **Grep before you read**: locate the exact symbol/section with a search tool first; read only
  the relevant line range instead of whole files (many transformer files include large
  docstring `Examples` blocks that aren't needed to understand the code).
- **Scope test runs**: target the specific `tests/test_*.py::test_name` under change while
  iterating; run the full `pytest` suite only once before finishing, not after every edit.
- **Don't paste large blocks back**: when confirming an edit, summarize what changed instead of
  re-printing the full function/file in chat.
- **Batch independent edits** into one multi-replace call rather than one tool call per file.
- **Reuse existing patterns**: this codebase has one canonical implementation per concern
  (single `_base_transformer.py`, one converter file per module) — check there first instead of
  re-deriving conventions from scratch across multiple files.
- **Prefer `--no-cov`** for quick local test iterations; coverage/HTML/XML report generation is
  only needed for a final verification pass.

---

**End of CLAUDE.md** — see [CONTRIBUTING.md](CONTRIBUTING.md) for questions or updates.
