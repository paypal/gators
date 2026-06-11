"""Tests for PSIFilter."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_selection import PSIFilter


@pytest.fixture
def reference_df():
    return pl.DataFrame(
        {
            "stable": [float(i % 10) for i in range(200)],
            "drifted": [float(i) for i in range(200)],
            "cat": ["a", "b"] * 100,
        }
    )


@pytest.fixture
def current_stable(reference_df):
    """Distribution identical to reference → PSI ≈ 0."""
    return pl.DataFrame(
        {
            "stable": [float(i % 10) for i in range(200)],
            "drifted": [float(i) for i in range(200)],
            "cat": ["a", "b"] * 100,
        }
    )


@pytest.fixture
def current_drifted(reference_df):
    """drifted column shifted far from reference → high PSI."""
    return pl.DataFrame(
        {
            "stable": [float(i % 10) for i in range(200)],
            "drifted": [float(i + 10_000) for i in range(200)],
            "cat": ["a", "b"] * 100,
        }
    )


class TestPSIFilter:
    def test_basic_fit_returns_self(self, reference_df, current_stable):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2)
        result = selector.fit(current_stable)
        assert result is selector

    def test_stable_column_kept(self, reference_df, current_stable):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2)
        selector.fit(current_stable)
        result = selector.transform(current_stable)
        assert "stable" in result.columns

    def test_drifted_column_dropped(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2)
        selector.fit(current_drifted)
        result = selector.transform(current_drifted)
        assert "drifted" not in result.columns

    def test_non_numeric_columns_always_kept(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2)
        selector.fit(current_drifted)
        result = selector.transform(current_drifted)
        assert "cat" in result.columns

    def test_zero_threshold_drops_any_drift(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=0.0)
        selector.fit(current_drifted)
        # With threshold=0, even tiny PSI > 0 gets dropped
        result = selector.transform(current_drifted)
        assert "drifted" not in result.columns

    def test_very_high_threshold_keeps_all(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=1e9)
        selector.fit(current_drifted)
        result = selector.transform(current_drifted)
        assert result.shape == current_drifted.shape

    def test_psi_scores_attribute(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2)
        selector.fit(current_drifted)
        assert "drifted" in selector.psi_scores_
        assert "stable" in selector.psi_scores_
        assert selector.psi_scores_["drifted"] > selector.psi_scores_["stable"]

    def test_selected_and_dropped_partition(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2)
        selector.fit(current_drifted)
        all_cols = set(current_drifted.columns)
        assert set(selector.selected_features_) | set(selector.columns_to_drop_) == all_cols
        assert set(selector.selected_features_) & set(selector.columns_to_drop_) == set()

    def test_subset_parameter(self, reference_df, current_drifted):
        selector = PSIFilter(reference_df=reference_df, threshold=0.2, subset=["stable"])
        selector.fit(current_drifted)
        # Only 'stable' evaluated; 'drifted' not in subset → not dropped
        assert "drifted" in selector.transform(current_drifted).columns

    def test_get_params(self, reference_df):
        selector = PSIFilter(reference_df=reference_df, threshold=0.15, n_bins=5)
        params = selector.get_params()
        assert params["threshold"] == 0.15
        assert params["n_bins"] == 5

    def test_set_params(self, reference_df):
        selector = PSIFilter(reference_df=reference_df)
        selector.set_params(threshold=0.3)
        assert selector.threshold == 0.3

    def test_identical_distributions_near_zero_psi(self, reference_df, current_stable):
        selector = PSIFilter(reference_df=reference_df, threshold=0.1)
        selector.fit(current_stable)
        # Both columns have identical distributions → PSI should be near 0
        assert selector.psi_scores_["stable"] < 0.01
