"""Tests for PermutationImportanceSelector."""

import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from gators.feature_selection import PermutationImportanceSelector


@pytest.fixture
def sample_data():
    X = pl.DataFrame(
        {
            "informative": [i % 2 for i in range(100)],
            "noise": [0] * 100,
            "semi": [(i // 5) % 2 for i in range(100)],
        }
    )
    y = pl.Series("target", [i % 2 for i in range(100)])
    return X, y


@pytest.fixture
def estimator():
    return RandomForestClassifier(n_estimators=20, random_state=0)


class TestPermutationImportanceSelector:
    def test_basic_fit_returns_self(self, sample_data, estimator):
        X, y = sample_data
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3, threshold=0.0)
        result = selector.fit(X, y)
        assert result is selector

    def test_drops_zero_importance_columns(self, sample_data, estimator):
        X, y = sample_data
        # A small positive threshold (> 0) drops the constant-noise column whose
        # permutation importance is exactly 0.0 (permuting it changes nothing).
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=5, threshold=1e-6)
        selector.fit(X, y)
        result = selector.transform(X)
        # constant noise column should have zero permutation importance → dropped
        assert "noise" not in result.columns

    def test_keeps_all_at_very_low_threshold(self, sample_data, estimator):
        X, y = sample_data
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3, threshold=-1.0)
        selector.fit(X, y)
        result = selector.transform(X)
        assert set(result.columns) == set(X.columns)

    def test_importances_attribute_populated(self, sample_data, estimator):
        X, y = sample_data
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3, threshold=0.0)
        selector.fit(X, y)
        assert set(selector.importances_.keys()) == set(X.columns)

    def test_selected_and_dropped_partition(self, sample_data, estimator):
        X, y = sample_data
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3, threshold=0.0)
        selector.fit(X, y)
        assert set(selector.selected_features_) | set(selector.columns_to_drop_) == set(X.columns)
        assert set(selector.selected_features_) & set(selector.columns_to_drop_) == set()

    def test_requires_y(self, sample_data, estimator):
        X, _ = sample_data
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3)
        with pytest.raises(ValueError, match="y must be provided"):
            selector.fit(X)

    def test_get_params(self, estimator):
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=7, threshold=0.01)
        params = selector.get_params()
        assert params["n_repeats"] == 7
        assert params["threshold"] == 0.01

    def test_set_params(self, estimator):
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3)
        selector.set_params(threshold=0.05)
        assert selector.threshold == 0.05

    def test_no_drop_when_empty_drop_list(self, sample_data, estimator):
        X, y = sample_data
        selector = PermutationImportanceSelector(estimator=estimator, n_repeats=3, threshold=-999.0)
        selector.fit(X, y)
        result = selector.transform(X)
        assert result.shape == X.shape
