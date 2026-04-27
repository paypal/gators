# Gators Documentation

This directory contains the Sphinx documentation for the gators package.

## Documentation Structure

The documentation is organized into three main sections:

### Getting Started
- **Installation** - How to install and verify the installation
- **Quick Start** - Get started in minutes with examples

### User Guide
- **Data Cleaning** - Quality filters, deduplication, outlier detection
- **Categorical Encoding** - OneHot, Target, WOE, and more
- **Feature Generation** - Numeric, string, and datetime feature engineering
- **Missing Value Imputation** - Handle missing data intelligently
- **Feature Scaling** - Normalize and standardize features
- **Pipeline** - Chain transformers together

### API Reference
- **Consolidated API Reference** - All transformers organized by category with direct links

## Building the Documentation

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install sphinx pydata-sphinx-theme sphinx-autodoc-typehints
```

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
python3.14 -m sphinx -M html source build
```

Or simply use:

```bash
cd docs
make html
```

The generated HTML documentation will be in `docs/build/html/index.html`.

### Viewing the Documentation

Open the documentation in your browser:

```bash
open build/html/index.html
```

### Rebuilding After Code Changes

If you've updated docstrings or added new modules, regenerate the API documentation:

```bash
cd docs
python3.14 -m sphinx.ext.apidoc -o source/ ../gators --force --separate
python3.14 -m sphinx -M html source build
```

### Cleaning Build Files

To remove generated documentation and start fresh:

```bash
cd docs
rm -rf build/
python3.14 -m sphinx -M html source build
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main page
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst       # Quick start guide
│   ├── user_guide/          # User guides
│   │   ├── data_cleaning.rst
│   │   ├── encoding.rst
│   │   ├── feature_generation.rst
│   │   ├── imputation.rst
│   │   ├── scaling.rst
│   │   └── pipeline.rst
│   ├── modules.rst          # API modules
│   ├── gators.*.rst         # Auto-generated API docs
│   ├── _static/             # Static files
│   └── _templates/          # Template overrides
└── build/                   # Generated documentation (git-ignored)
```

## Theme

The documentation uses the **PyData Sphinx Theme**, which is the same theme used by:
- NumPy
- Pandas
- Matplotlib
- SciPy

Features:
- Modern, clean design
- Responsive layout for mobile
- Organized sidebar navigation
- Built-in search functionality
- Light/dark mode support

## Warnings

The build process may show warnings about docstring formatting. These are non-critical and don't prevent documentation generation. To improve documentation quality, consider:

1. Using proper reStructuredText formatting in docstrings
2. Adding blank lines between sections
3. Using backticks for inline code references
4. Proper indentation in parameter descriptions
