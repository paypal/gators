"""Tests for MutualInformationSelector — 100 % branch coverage."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_selection import MutualInformationSelector


@pytest.fixture
def binary_data():
    """Perfectly correlated + noise + categorical features with binary target."""
    X = pl.DataFrame(
        {
            "perfect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "noise": [8.0, 1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    y = pl.Series("target", [0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


class TestMutualInformationSelector:
    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.0)
        assert sel.fit(X, y) is sel

    def test_y_required_raises(self, binary_data):
        X, _ = binary_data
        with pytest.raises(ValueError, match="y must be provided"):
            MutualInformationSelector(threshold=0.0).fit(X)

    def test_zero_threshold_keeps_all(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.columns == X.columns

    def test_high_threshold_drops_noise(self):
        X = pl.DataFrame(
            {
                "informative": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        y = pl.Series("target", [0, 1, 0, 1, 0, 1, 0, 1])
        sel = MutualInformationSelector(threshold=0.1)
        sel.fit(X, y)
        result = sel.transform(X)
        assert "informative" in result.columns
        assert "constant" not in result.columns

    def test_mi_values_attribute(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        assert set(sel.mi_values_.keys()) == set(X.columns)
        assert all(v >= 0 for v in sel.mi_values_.values())

    def test_selected_and_dropped_attributes(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.5)
        sel.fit(X, y)
        assert set(sel.selected_features_) | set(sel.columns_to_drop_) == set(X.columns)
        assert not (set(sel.selected_features_) & set(sel.columns_to_drop_))

    def test_subset_parameter_restricts_evaluation(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.0, subset=["perfect"])
        sel.fit(X, y)
        # Only "perfect" was scored; noise and cat are always kept
        assert "noise" in sel.selected_features_
        assert "cat" in sel.selected_features_

    def test_numeric_only_dataframe(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        y = pl.Series("t", [0, 1, 0, 1])
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        result = sel.transform(X)
        assert_frame_equal(result, X)

    def test_categorical_only_dataframe(self):
        X = pl.DataFrame({"c": ["a", "b", "a", "b", "a", "b"]})
        y = pl.Series("t", [0, 1, 0, 1, 0, 1])
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        assert sel.mi_values_["c"] > 0

    def test_constant_column_gives_zero_mi(self):
        X = pl.DataFrame({"const": [1.0, 1.0, 1.0, 1.0], "other": [1.0, 2.0, 3.0, 4.0]})
        y = pl.Series("t", [0, 1, 0, 1])
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        assert sel.mi_values_["const"] == pytest.approx(0.0)

    def test_all_nulls_column(self):
        X = pl.DataFrame({"all_null": [None, None, None, None], "good": [1.0, 2.0, 3.0, 4.0]})
        y = pl.Series("t", [0, 1, 0, 1])
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        # No data after drop_nulls → MI = 0
        assert sel.mi_values_["all_null"] == pytest.approx(0.0)

    def test_boolean_column(self):
        X = pl.DataFrame({"flag": [True, False, True, False, True, False]})
        y = pl.Series("t", [1, 0, 1, 0, 1, 0])
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        assert sel.mi_values_["flag"] > 0

    def test_n_bins_parameter(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.0, n_bins=5)
        sel.fit(X, y)
        assert all(v >= 0 for v in sel.mi_values_.values())

    def test_get_params(self):
        sel = MutualInformationSelector(threshold=0.1, n_bins=5)
        params = sel.get_params()
        assert params["threshold"] == 0.1
        assert params["n_bins"] == 5

    def test_set_params(self):
        sel = MutualInformationSelector(threshold=0.0)
        sel.set_params(threshold=0.3)
        assert sel.threshold == 0.3

    def test_fit_transform(self, binary_data):
        X, y = binary_data
        sel = MutualInformationSelector(threshold=0.0)
        result = sel.fit_transform(X, y)
        assert isinstance(result, pl.DataFrame)

    def test_all_null_y_gives_zero_mi(self):
        """When all target values are null, n==0 after drop_nulls → MI = 0."""
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pl.Series("t", [None, None, None], dtype=pl.Int64)
        sel = MutualInformationSelector(threshold=0.0)
        sel.fit(X, y)
        assert sel.mi_values_["a"] == pytest.approx(0.0)
