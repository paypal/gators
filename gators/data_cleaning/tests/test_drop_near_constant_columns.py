"""Tests for DropNearConstantColumns transformer."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.data_cleaning import DropNearConstantColumns


class TestDropNearConstantColumns:
    """Test suite for DropNearConstantColumns."""

    # ------------------------------------------------------------------
    # Basic drop behaviour
    # ------------------------------------------------------------------

    def test_drop_near_constant_numeric(self):
        """Column with 1 unique value out of 100 rows is dropped at threshold=0.02."""
        X = pl.DataFrame(
            {
                "id": list(range(100)),
                "near_const": [42] * 99 + [0],
                "varying": list(range(100)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.02)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"id": list(range(100)), "varying": list(range(100))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["near_const"]

    def test_drop_near_constant_categorical(self):
        """Near-constant categorical column is dropped."""
        X = pl.DataFrame(
            {
                "country": ["USA"] * 9 + ["UK"],
                "city": [
                    "NYC",
                    "LA",
                    "Chicago",
                    "Boston",
                    "Seattle",
                    "Denver",
                    "Miami",
                    "Austin",
                    "Portland",
                    "Dallas",
                ],
            }
        )
        remover = DropNearConstantColumns(threshold=0.25)
        result = remover.fit_transform(X)

        expected = pl.DataFrame(
            {
                "city": [
                    "NYC",
                    "LA",
                    "Chicago",
                    "Boston",
                    "Seattle",
                    "Denver",
                    "Miami",
                    "Austin",
                    "Portland",
                    "Dallas",
                ]
            }
        )
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["country"]

    def test_no_near_constant_columns(self):
        """No columns dropped when all columns are sufficiently diverse."""
        X = pl.DataFrame(
            {
                "col1": list(range(10)),
                "col2": [str(i) for i in range(10)],
            }
        )
        remover = DropNearConstantColumns(threshold=0.05)
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_drop_multiple_near_constant_columns(self):
        """Multiple near-constant columns are all dropped."""
        X = pl.DataFrame(
            {
                "nc1": [1] * 9 + [2],
                "nc2": ["A"] * 9 + ["B"],
                "varying": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.25)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"nc1", "nc2"}

    def test_all_columns_near_constant(self):
        """All columns dropped yields an empty-column DataFrame."""
        X = pl.DataFrame({"nc1": [1] * 9 + [2], "nc2": ["A"] * 9 + ["B"]})
        remover = DropNearConstantColumns(threshold=0.25)
        result = remover.fit_transform(X)

        assert result.shape[1] == 0
        assert set(remover._to_drop) == {"nc1", "nc2"}

    # ------------------------------------------------------------------
    # Threshold boundary
    # ------------------------------------------------------------------

    def test_column_exactly_at_threshold_is_dropped(self):
        """Column with n_unique == threshold * n_rows is dropped (<=)."""
        # 10 rows, threshold=0.1 → max_unique=1.0; column has 1 unique value
        X = pl.DataFrame({"nc": [42] * 10, "other": list(range(10))})
        remover = DropNearConstantColumns(threshold=0.1)
        result = remover.fit_transform(X)

        assert "nc" not in result.columns
        assert remover._to_drop == ["nc"]

    def test_column_just_above_threshold_is_kept(self):
        """Column with n_unique just above threshold * n_rows is kept."""
        # 10 rows, threshold=0.1 → max_unique=1.0; column has 2 unique values → kept
        X = pl.DataFrame({"nc": [42] * 9 + [0], "other": list(range(10))})
        remover = DropNearConstantColumns(threshold=0.1)
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_threshold_zero_behaves_like_drop_constant(self):
        """threshold=0 drops only truly constant columns (n_unique <= 0 is impossible
        for non-empty columns, so only empty columns / all-same columns are dropped)."""
        X = pl.DataFrame(
            {
                "const": [1, 1, 1],
                "near_const": [1, 1, 2],
                "varying": [1, 2, 3],
            }
        )
        remover = DropNearConstantColumns(threshold=0.0)
        result = remover.fit_transform(X)

        # threshold=0 → max_unique=0 → only columns with n_unique<=0 dropped
        # None of these columns have n_unique<=0, so nothing is dropped
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    # ------------------------------------------------------------------
    # include_na parameter
    # ------------------------------------------------------------------

    def test_include_na_true_null_counts_as_unique(self):
        """With include_na=True nulls count as a distinct value."""
        # 10 rows; 'mostly_null' has 2 unique values (None + 1) but only 2 → <= 0.15*10=1.5 → dropped
        X = pl.DataFrame(
            {
                "mostly_null": [None] * 9 + [1],
                "varying": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.25, include_na=True)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["mostly_null"]

    def test_include_na_false_null_ignored(self):
        """With include_na=False nulls are excluded before counting."""
        # 'same_non_null': non-null values are all 1 → 1 unique (excl. null) → dropped
        X = pl.DataFrame(
            {
                "same_non_null": [1] * 9 + [None],
                "varying": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.15, include_na=False)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["same_non_null"]

    def test_include_na_false_null_adds_no_unique(self):
        """Column with two real values plus nulls is kept when include_na=False."""
        X = pl.DataFrame(
            {
                "two_vals": [1, 2, None, None, None, None, None, None, None, None],
                "varying": list(range(10)),
            }
        )
        # threshold=0.15 → max_unique=1.5; two_vals has 2 non-null unique values → kept
        remover = DropNearConstantColumns(threshold=0.15, include_na=False)
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)
        assert remover._to_drop == []

    # ------------------------------------------------------------------
    # subset parameter
    # ------------------------------------------------------------------

    def test_subset_only_checks_specified_columns(self):
        """Only columns listed in subset are candidates for dropping."""
        X = pl.DataFrame(
            {
                "col1": [1] * 9 + [2],
                "col2": [5] * 9 + [6],
                "col3": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.25, subset=["col1", "col2"])
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"col3": list(range(10))})
        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"col1", "col2"}

    def test_subset_column_not_near_constant_is_kept(self):
        """A subset column that is not near-constant is kept."""
        X = pl.DataFrame(
            {
                "col1": [1] * 9 + [2],
                "col2": list(range(10)),
                "col3": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.25, subset=["col1", "col2"])
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"col2": list(range(10)), "col3": list(range(10))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["col1"]

    def test_subset_does_not_check_non_subset_columns(self):
        """Near-constant columns outside subset are not dropped."""
        X = pl.DataFrame(
            {
                "nc_outside": [1] * 9 + [2],  # near-constant but outside subset
                "col2": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(threshold=0.15, subset=["col2"])
        result = remover.fit_transform(X)

        # nc_outside is not in subset → not dropped
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    # ------------------------------------------------------------------
    # fit / transform split (train / test consistency)
    # ------------------------------------------------------------------

    def test_fit_transform_separately(self):
        """Fitted transformer drops same columns on unseen data."""
        train = pl.DataFrame({"nc": [1] * 9 + [2], "varying": list(range(10))})
        test = pl.DataFrame({"nc": list(range(5)), "varying": list(range(5))})

        remover = DropNearConstantColumns(threshold=0.25)
        remover.fit(train)
        result = remover.transform(test)

        expected = pl.DataFrame({"varying": list(range(5))})
        assert_frame_equal(result, expected)

    def test_transform_uses_fitted_columns_not_new_data(self):
        """transform() honours fit-time decision even if column changed in test data."""
        train = pl.DataFrame({"nc": [1] * 9 + [2], "varying": list(range(10))})
        # In test data 'nc' is now fully varying
        test = pl.DataFrame({"nc": list(range(10)), "varying": list(range(10))})

        remover = DropNearConstantColumns(threshold=0.25)
        remover.fit(train)
        result = remover.transform(test)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_row_dataframe(self):
        """With one row every column has 1 unique value."""
        X = pl.DataFrame({"col1": [1], "col2": ["A"]})
        # threshold=0.5 → max_unique=0.5; n_unique=1 > 0.5 → kept
        remover = DropNearConstantColumns(threshold=0.5)
        result = remover.fit_transform(X)
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_empty_dataframe(self):
        """Empty DataFrame (zero rows) produces zero unique values → all dropped."""
        X = pl.DataFrame({"col1": [], "col2": []}).cast({"col1": pl.Int64, "col2": pl.String})
        remover = DropNearConstantColumns(threshold=0.01)
        result = remover.fit_transform(X)
        # 0 <= 0.01*0=0 → all dropped
        assert result.shape[1] == 0

    def test_boolean_near_constant_column(self):
        """Boolean columns are handled correctly."""
        X = pl.DataFrame(
            {
                "mostly_true": [True] * 9 + [False],
                "mixed": [True, False] * 5,
            }
        )
        # threshold=0.15 → max_unique=1.5; mostly_true has 2 unique → kept
        remover = DropNearConstantColumns(threshold=0.15)
        result = remover.fit_transform(X)
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_boolean_constant_dropped_at_low_threshold(self):
        """A boolean column with 1 unique value is dropped."""
        X = pl.DataFrame(
            {
                "all_true": [True] * 10,
                "mixed": [True, False] * 5,
            }
        )
        remover = DropNearConstantColumns(threshold=0.15)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"mixed": [True, False] * 5})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["all_true"]

    # ------------------------------------------------------------------
    # Sklearn API compatibility
    # ------------------------------------------------------------------

    def test_get_params(self):
        """get_params() returns correct parameter dict."""
        remover = DropNearConstantColumns(threshold=0.05, subset=["a"], include_na=False)
        params = remover.get_params()
        assert params["threshold"] == 0.05
        assert params["subset"] == ["a"]
        assert params["include_na"] is False

    def test_set_params(self):
        """set_params() updates parameters correctly."""
        remover = DropNearConstantColumns()
        remover.set_params(threshold=0.05, include_na=False)
        assert remover.threshold == 0.05
        assert remover.include_na is False
