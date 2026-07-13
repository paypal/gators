"""Tests for CorrelationSelector."""

import pytest
import polars as pl
from polars.testing import assert_frame_equal

from gators.feature_selection import CorrelationSelector


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.1, 2.1, 3.1, 4.1, 5.1],  # highly correlated with "a"
            "c": [5.0, 3.0, 1.0, 4.0, 2.0],  # independent
        }
    )


@pytest.fixture
def importance():
    return {"a": 0.9, "b": 0.4, "c": 0.7}


class TestCorrelationSelector:
    def test_basic_fit_returns_self(self, sample_df, importance):
        selector = CorrelationSelector(importance=importance)
        result = selector.fit(sample_df)
        assert result is selector

    def test_drops_lower_importance_correlated_column(self, sample_df, importance):
        selector = CorrelationSelector(importance=importance, max_corr=0.95)
        selector.fit(sample_df)
        assert "b" in selector.columns_to_drop_
        assert "a" in selector.selected_features_

    def test_transform_removes_dropped_columns(self, sample_df, importance):
        selector = CorrelationSelector(importance=importance)
        selector.fit(sample_df)
        result = selector.transform(sample_df)
        assert "b" not in result.columns
        assert "a" in result.columns
        assert "c" in result.columns

    def test_selected_and_dropped_partition(self, sample_df, importance):
        selector = CorrelationSelector(importance=importance)
        selector.fit(sample_df)
        all_cols = set(sample_df.columns)
        assert set(selector.selected_features_) | set(selector.columns_to_drop_) == all_cols
        assert set(selector.selected_features_) & set(selector.columns_to_drop_) == set()

    def test_strict_threshold_keeps_all(self, sample_df, importance):
        """max_corr=1.0 only removes perfectly correlated pairs (rare in floats)."""
        selector = CorrelationSelector(importance=importance, max_corr=1.0)
        selector.fit(sample_df)
        result = selector.transform(sample_df)
        assert result.columns == sample_df.columns

    def test_no_drop_when_columns_to_drop_empty(self, sample_df, importance):
        selector = CorrelationSelector(importance=importance, max_corr=1.0)
        selector.fit(sample_df)
        result = selector.transform(sample_df)
        assert_frame_equal(result, sample_df)

    def test_invalid_max_corr_zero_raises(self, importance):
        with pytest.raises(Exception):
            CorrelationSelector(importance=importance, max_corr=0.0)

    def test_invalid_max_corr_above_one_raises(self, importance):
        with pytest.raises(Exception):
            CorrelationSelector(importance=importance, max_corr=1.1)

    def test_use_abs_false_ignores_negative_correlations(self):
        """Strong negative correlation should NOT trigger removal when use_abs=False."""
        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [-1.0, -2.0, -3.0, -4.0, -5.0],
                "c": [5.0, 3.0, 1.0, 4.0, 2.0],
            }
        )
        imp = {"a": 0.9, "b": 0.4, "c": 0.7}
        selector = CorrelationSelector(importance=imp, max_corr=0.95, use_abs=False)
        selector.fit(X)
        assert selector.columns_to_drop_ == []

    def test_col_i_dropped_when_lower_importance(self):
        """When col_i is less important than col_j, col_i is dropped and loop breaks."""
        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [1.1, 2.1, 3.1, 4.1, 5.1],
                "c": [5.0, 3.0, 1.0, 4.0, 2.0],
            }
        )
        imp = {"a": 0.4, "b": 0.9, "c": 0.7}  # b is more important than a
        selector = CorrelationSelector(importance=imp, max_corr=0.95)
        selector.fit(X)
        assert "a" in selector.columns_to_drop_
        assert "b" in selector.selected_features_

    def test_single_candidate_no_pairs_no_drop(self):
        """Only one column listed in importance → no pairs → nothing dropped."""
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        selector = CorrelationSelector(importance={"a": 0.9}, max_corr=0.95)
        selector.fit(X)
        assert selector.columns_to_drop_ == []

    def test_non_numeric_column_never_candidate(self):
        """String columns are skipped even if present in importance."""
        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [1.1, 2.1, 3.1, 4.1, 5.1],
                "cat": ["x", "y", "x", "y", "x"],
            }
        )
        imp = {"a": 0.9, "b": 0.4, "cat": 0.1}
        selector = CorrelationSelector(importance=imp)
        selector.fit(X)
        result = selector.transform(X)
        assert "cat" in result.columns

    def test_constant_column_nan_correlation_skipped(self):
        """Constant column yields null correlation → skipped, not dropped."""
        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "const": [1.0, 1.0, 1.0, 1.0, 1.0],
                "c": [5.0, 3.0, 1.0, 4.0, 2.0],
            }
        )
        imp = {"a": 0.9, "const": 0.4, "c": 0.7}
        selector = CorrelationSelector(importance=imp)
        selector.fit(X)
        assert "const" in selector.selected_features_

    def test_column_not_in_importance_always_kept(self):
        """Columns absent from importance are never candidates and always survive."""
        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [1.1, 2.1, 3.1, 4.1, 5.1],
                "extra": [9.0, 8.0, 7.0, 6.0, 5.0],
            }
        )
        imp = {"a": 0.9, "b": 0.4}  # "extra" not listed
        selector = CorrelationSelector(importance=imp)
        selector.fit(X)
        result = selector.transform(X)
        assert "extra" in result.columns

    def test_inner_continue_col_j_already_dropped(self):
        """col_j already in columns_to_drop from a prior pair → inner continue hit."""
        # a (i=0) and c (i=2) are correlated; a is more important → c dropped.
        # When the outer loop reaches b (i=1) and tries j=2 (c), c is already
        # in columns_to_drop → the inner `continue` branch is exercised.
        X = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 3.0, 1.0, 4.0, 2.0],  # independent
                "c": [1.1, 2.1, 3.1, 4.1, 5.1],  # highly correlated with a → dropped
                "d": [5.1, 3.1, 1.1, 4.1, 2.1],  # independent
            }
        )
        imp = {"a": 0.9, "b": 0.7, "c": 0.4, "d": 0.6}
        selector = CorrelationSelector(importance=imp, max_corr=0.95)
        selector.fit(X)
        assert "c" in selector.columns_to_drop_
        assert "a" in selector.selected_features_

    def test_get_params(self, importance):
        selector = CorrelationSelector(importance=importance, max_corr=0.8, use_abs=False)
        params = selector.get_params()
        assert params["max_corr"] == 0.8
        assert params["use_abs"] is False

    def test_set_params(self, importance):
        selector = CorrelationSelector(importance=importance)
        selector.set_params(max_corr=0.7)
        assert selector.max_corr == 0.7
