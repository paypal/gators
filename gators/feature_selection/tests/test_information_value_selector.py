"""Tests for InformationValueSelector."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_selection import InformationValueSelector


@pytest.fixture
def sample_data():
    X = pl.DataFrame(
        {
            "cat_strong": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "cat_weak": ["x", "x", "x", "x", "x", "x", "y", "y"],
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    y = pl.Series("target", [1, 0, 1, 0, 1, 0, 1, 0])
    return X, y


class TestInformationValueSelector:
    def test_basic_fit_returns_self(self, sample_data):
        X, y = sample_data
        selector = InformationValueSelector(threshold=0.02)
        result = selector.fit(X, y)
        assert result is selector

    def test_drops_low_iv_columns(self, sample_data):
        X, y = sample_data
        # cat_weak has very low IV; set a moderate threshold to ensure it's dropped
        selector = InformationValueSelector(threshold=0.5)
        selector.fit(X, y)
        result = selector.transform(X)
        # numeric column is always kept (not evaluated for IV)
        assert "numeric" in result.columns
        # at least cat_weak should be dropped at a high threshold
        assert len(result.columns) < len(X.columns)

    def test_keeps_all_if_threshold_zero(self, sample_data):
        X, y = sample_data
        selector = InformationValueSelector(threshold=0.0)
        selector.fit(X, y)
        result = selector.transform(X)
        assert result.columns == X.columns

    def test_numeric_only_dataframe(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pl.Series("target", [1, 0, 1])
        selector = InformationValueSelector(threshold=0.02)
        selector.fit(X, y)
        result = selector.transform(X)
        # No categorical columns → nothing to drop
        assert_frame_equal(result, X)

    def test_selected_and_dropped_attributes(self, sample_data):
        X, y = sample_data
        selector = InformationValueSelector(threshold=0.5)
        selector.fit(X, y)
        assert set(selector.selected_features_) | set(selector.columns_to_drop_) == set(X.columns)
        assert set(selector.selected_features_) & set(selector.columns_to_drop_) == set()

    def test_requires_y(self, sample_data):
        X, _ = sample_data
        selector = InformationValueSelector()
        with pytest.raises(ValueError, match="y must be provided"):
            selector.fit(X)

    def test_get_params(self):
        selector = InformationValueSelector(threshold=0.05, regularization=0.001)
        params = selector.get_params()
        assert params["threshold"] == 0.05
        assert params["regularization"] == 0.001

    def test_set_params(self):
        selector = InformationValueSelector()
        selector.set_params(threshold=0.1)
        assert selector.threshold == 0.1

    def test_transform_no_drop_when_empty_drop_list(self, sample_data):
        X, y = sample_data
        selector = InformationValueSelector(threshold=0.0)
        selector.fit(X, y)
        result = selector.transform(X)
        assert result.shape == X.shape
