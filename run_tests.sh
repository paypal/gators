#!/bin/bash
# Run tests with coverage
python3.14 -m pytest --cov=gators --cov-report=term --cov-report=html

# Generate coverage badge
python3.14 -m coverage_badge -o docs/source/_static/coverage.svg -f