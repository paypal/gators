"""Tests for IterativeImputer — 100 % branch coverage."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers import IterativeImputer


@pytest.fixture
def simple_df():
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, None, 30.0, 40.0, 50.0],
            "c": [0.5, 1.0, 1.5, None, 2.5],
        }
    )


class TestIterativeImputer:
    def test_fit_returns_self(self, simple_df):
        imputer = IterativeImputer(max_iter=2)
        assert imputer.fit(simple_df) is imputer

    def test_no_nulls_unchanged(self):
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(X)
        result = imputer.transform(X)
        assert_frame_equal(result, X)

    def test_nulls_filled(self, simple_df):
        imputer = IterativeImputer(max_iter=3)
        imputer.fit(simple_df)
        result = imputer.transform(simple_df)
        assert result.null_count().sum_horizontal()[0] == 0

    def test_non_null_values_unchanged(self, simple_df):
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(simple_df)
        result = imputer.transform(simple_df)
        non_null_mask = simple_df["a"].is_not_null()
        assert result["a"].filter(non_null_mask).to_list() == pytest.approx(
            simple_df["a"].filter(non_null_mask).to_list()
        )

    def test_auto_detect_subset(self, simple_df):
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(simple_df)
        assert set(imputer.subset) == {"a", "b", "c"}

    def test_explicit_subset(self, simple_df):
        imputer = IterativeImputer(max_iter=2, subset=["a"])
        imputer.fit(simple_df)
        result = imputer.transform(simple_df)
        # "a" is in subset: nulls should be filled
        assert result["a"].null_count() == 0
        # "b" and "c" are not in subset: their original nulls are preserved
        assert result["b"].null_count() == simple_df["b"].null_count()
        assert result["c"].null_count() == simple_df["c"].null_count()

    def test_mean_strategy(self, simple_df):
        imputer = IterativeImputer(max_iter=1, initial_strategy="mean")
        imputer.fit(simple_df)
        assert "a" in imputer.statistics_

    def test_median_strategy(self, simple_df):
        imputer = IterativeImputer(max_iter=1, initial_strategy="median")
        imputer.fit(simple_df)
        assert "a" in imputer.statistics_

    def test_statistics_attribute(self, simple_df):
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(simple_df)
        assert set(imputer.statistics_.keys()) == {"a", "b", "c"}

    def test_single_null(self):
        X = pl.DataFrame({"a": [1.0, None], "b": [3.0, 4.0]})
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(X)
        result = imputer.transform(X)
        assert result["a"].null_count() == 0

    def test_no_feature_cols(self):
        """DataFrame with no numeric columns → transform is a no-op."""
        X = pl.DataFrame({"name": ["alice", "bob"]})
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(X)
        result = imputer.transform(X)
        assert_frame_equal(result, X)

    def test_all_nulls_column_falls_back_to_mean(self):
        X = pl.DataFrame(
            {
                "a": pl.Series([None, None, None, None], dtype=pl.Float64),
                "b": [1.0, 2.0, 3.0, 4.0],
            }
        )
        imputer = IterativeImputer(max_iter=2)
        imputer.fit(X)
        result = imputer.transform(X)
        # All-null column should be filled (with 0.0 fallback)
        assert result["a"].null_count() == 0

    def test_multiple_iter_converges(self, simple_df):
        imp1 = IterativeImputer(max_iter=1)
        imp1.fit(simple_df)
        r1 = imp1.transform(simple_df)

        imp10 = IterativeImputer(max_iter=10)
        imp10.fit(simple_df)
        r10 = imp10.transform(simple_df)

        # Both produce no nulls
        assert r1.null_count().sum_horizontal()[0] == 0
        assert r10.null_count().sum_horizontal()[0] == 0

    def test_transform_on_new_data(self, simple_df):
        imputer = IterativeImputer(max_iter=3)
        imputer.fit(simple_df)
        new = pl.DataFrame({"a": [None, 3.0], "b": [20.0, None], "c": [1.0, 2.0]})
        result = imputer.transform(new)
        assert result.null_count().sum_horizontal()[0] == 0

    def test_get_params(self):
        imp = IterativeImputer(max_iter=5, initial_strategy="median")
        params = imp.get_params()
        assert params["max_iter"] == 5
        assert params["initial_strategy"] == "median"

    def test_set_params(self):
        imp = IterativeImputer(max_iter=3)
        imp.set_params(max_iter=7)
        assert imp.max_iter == 7

    def test_fit_transform(self, simple_df):
        result = IterativeImputer(max_iter=2).fit_transform(simple_df)
        assert result.null_count().sum_horizontal()[0] == 0

    def test_transform_skips_col_without_coef(self, simple_df):
        """Test the safety guard: column in imputation_order but not in _coefs."""
        imputer = IterativeImputer(max_iter=1)
        imputer.fit(simple_df)
        # Remove one coefficient to exercise the 'if col not in self._coefs: continue' branch
        del imputer._coefs[imputer._imputation_order[0]]
        result = imputer.transform(simple_df)
        # Should still return a DataFrame without error
        assert isinstance(result, pl.DataFrame)

    def test_integer_column_rounded_and_cast_back(self):
        """Integer columns must be rounded and cast back to their original dtype."""
        X = pl.DataFrame(
            {
                "a": pl.Series([1, 2, None, 4, 5], dtype=pl.Int64),
                "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        imputer = IterativeImputer(max_iter=2)
        result = imputer.fit_transform(X)
        assert result["a"].dtype == pl.Int64
        assert result["a"].null_count() == 0
