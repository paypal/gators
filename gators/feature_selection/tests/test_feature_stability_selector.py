"""Tests for FeatureStabilitySelector."""

import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from gators.feature_selection import FeatureStabilitySelector


@pytest.fixture
def sample_data():
    pl.set_random_seed(42)
    X = pl.DataFrame(
        {
            "stable": [i % 2 for i in range(100)],
            "unstable": [i % 7 for i in range(100)],
            "noise": [0] * 100,
        }
    )
    y = pl.Series("target", [i % 2 for i in range(100)])
    return X, y


@pytest.fixture
def estimator():
    return RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)


@pytest.fixture
def skf():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


class TestFeatureStabilitySelector:
    def test_basic_fit_returns_self(self, sample_data, estimator, skf):
        X, y = sample_data
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.5)
        result = selector.fit(X, y)
        assert result is selector

    def test_drops_unstable_columns(self, sample_data, estimator, skf):
        X, y = sample_data
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.8)
        selector.fit(X, y)
        result = selector.transform(X)
        # noise (all zeros) should have zero/low FSI and be dropped
        assert "noise" not in result.columns

    def test_keeps_all_at_zero_threshold(self, sample_data, estimator, skf):
        # threshold=0.0 means keep features with FSI >= 0, i.e. everything
        # including constant-valued columns whose FSI is exactly 0.
        X, y = sample_data
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.0)
        selector.fit(X, y)
        result = selector.transform(X)
        # All columns must survive because none has FSI < 0
        assert set(result.columns) == set(X.columns)

    def test_selected_and_dropped_attributes_partition(self, sample_data, estimator, skf):
        X, y = sample_data
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.5)
        selector.fit(X, y)
        assert set(selector.selected_features_) | set(selector.columns_to_drop_) == set(X.columns)
        assert set(selector.selected_features_) & set(selector.columns_to_drop_) == set()

    def test_fsi_scores_attribute(self, sample_data, estimator, skf):
        X, y = sample_data
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.5)
        selector.fit(X, y)
        assert selector.fsi_scores_.columns == ["feature", "fsi", "importance"]

    def test_requires_y(self, sample_data, estimator, skf):
        X, _ = sample_data
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf)
        with pytest.raises(ValueError, match="y must be provided"):
            selector.fit(X)

    def test_get_params_contains_keys(self, estimator, skf):
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.6)
        params = selector.get_params()
        assert "threshold" in params
        assert params["threshold"] == 0.6

    def test_set_params(self, estimator, skf):
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf)
        selector.set_params(threshold=0.9)
        assert selector.threshold == 0.9

    def test_transform_no_drop_when_empty(self, sample_data, estimator, skf):
        X, y = sample_data
        # threshold=0.0 → nothing is dropped, shape must be preserved
        selector = FeatureStabilitySelector(estimator=estimator, skf=skf, threshold=0.0)
        selector.fit(X, y)
        result = selector.transform(X)
        assert result.shape == X.shape
