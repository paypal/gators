# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-03-13

### Added
- Initial public release of Gators
- 60+ preprocessing transformers across multiple categories:
  - Data cleaning (13 transformers)
  - Categorical encoding (9 encoders)
  - Feature generation for numeric, string, and datetime data (29 generators)
  - Missing value imputation (4 imputers)
  - Discretization (6 discretizers)
  - Feature scaling (3 scalers)
- Pipeline support for chaining transformers
- Comprehensive test suite with 867 tests and 87% coverage
- CI/CD with GitHub Actions
- MIT License

[Unreleased]: https://github.com/paypal/gators/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/paypal/gators/releases/tag/v1.0.1
