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
        """Column whose top value covers 99% of rows is dropped at max_ratio=0.98."""
        X = pl.DataFrame(
            {
                "id": list(range(100)),
                "near_const": [42] * 99 + [0],
                "varying": list(range(100)),
            }
        )
        remover = DropNearConstantColumns(max_ratio=0.98)
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
        remover = DropNearConstantColumns(max_ratio=0.8)
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
        """No columns dropped when no value dominates."""
        X = pl.DataFrame(
            {
                "col1": list(range(10)),
                "col2": [str(i) for i in range(10)],
            }
        )
        remover = DropNearConstantColumns(max_ratio=0.5)
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
        remover = DropNearConstantColumns(max_ratio=0.8)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert set(remover._to_drop) == {"nc1", "nc2"}

    def test_all_columns_near_constant(self):
        """All columns dropped yields an empty-column DataFrame."""
        X = pl.DataFrame({"nc1": [1] * 9 + [2], "nc2": ["A"] * 9 + ["B"]})
        remover = DropNearConstantColumns(max_ratio=0.8)
        result = remover.fit_transform(X)

        assert result.shape[1] == 0
        assert set(remover._to_drop) == {"nc1", "nc2"}

    # ------------------------------------------------------------------
    # max_ratio boundary
    # ------------------------------------------------------------------

    def test_column_exactly_at_max_ratio_is_dropped(self):
        """Column with mode share == max_ratio is dropped (>=)."""
        # 10 rows, all identical -> mode share = 1.0
        X = pl.DataFrame({"nc": [42] * 10, "other": list(range(10))})
        remover = DropNearConstantColumns(max_ratio=1.0)
        result = remover.fit_transform(X)

        assert "nc" not in result.columns
        assert remover._to_drop == ["nc"]

    def test_column_just_below_max_ratio_is_kept(self):
        """Column with mode share just below max_ratio is kept."""
        # 10 rows, 9 identical + 1 different -> mode share = 0.9
        X = pl.DataFrame({"nc": [42] * 9 + [0], "other": list(range(10))})
        remover = DropNearConstantColumns(max_ratio=0.95)
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_max_ratio_one_behaves_like_drop_constant(self):
        """max_ratio=1.0 only drops truly constant columns (mode share == 1.0)."""
        X = pl.DataFrame(
            {
                "const": [1, 1, 1],
                "near_const": [1, 1, 2],
                "varying": [1, 2, 3],
            }
        )
        remover = DropNearConstantColumns(max_ratio=1.0)
        result = remover.fit_transform(X)

        # const has mode share 1.0 -> dropped; near_const (2/3) and varying (1/3) kept
        expected = pl.DataFrame({"near_const": [1, 1, 2], "varying": [1, 2, 3]})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["const"]

    def test_max_ratio_one_matches_drop_constant_with_nulls(self):
        """max_ratio=1.0 with include_na=False exactly matches DropConstantColumns,
        even when a column is constant among non-nulls but also contains nulls."""
        X = pl.DataFrame(
            {
                "const_with_nulls": [1, 1, None, None, None],
                "varying": [1, 2, 3, 4, 5],
            }
        )
        remover = DropNearConstantColumns(max_ratio=1.0, include_na=False)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": [1, 2, 3, 4, 5]})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["const_with_nulls"]

    # ------------------------------------------------------------------
    # include_na parameter
    # ------------------------------------------------------------------

    def test_include_na_true_null_counts_as_dominant(self):
        """With include_na=True, null is treated as its own (possibly dominant) value."""
        # 10 rows; 'mostly_null' is null 9/10 times -> mode share 0.9
        X = pl.DataFrame(
            {
                "mostly_null": [None] * 9 + [1],
                "varying": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(max_ratio=0.8, include_na=True)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["mostly_null"]

    def test_include_na_false_null_ignored(self):
        """With include_na=False, nulls are excluded from both numerator and denominator."""
        # 'same_non_null': all 9 non-null values are 1 -> share among non-null rows = 1.0
        X = pl.DataFrame(
            {
                "same_non_null": [1] * 9 + [None],
                "varying": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(max_ratio=0.8, include_na=False)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["same_non_null"]

    def test_include_na_false_null_adds_no_dominant_share(self):
        """Column with two equally-common non-null values is kept when include_na=False."""
        X = pl.DataFrame(
            {
                "two_vals": [1, 2, None, None, None, None, None, None, None, None],
                "varying": list(range(10)),
            }
        )
        # non-null mode count = 1 out of 2 non-null values -> share = 0.5 -> kept at 0.6
        remover = DropNearConstantColumns(max_ratio=0.6, include_na=False)
        result = remover.fit_transform(X)

        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_include_na_false_all_null_column_is_dropped(self):
        """An entirely-null column has no non-null value at all, so it is treated
        as constant (dropped) when include_na=False, matching DropConstantColumns."""
        X = pl.DataFrame(
            {
                "all_null": pl.Series([None] * 10, dtype=pl.Int64),
                "varying": list(range(10)),
            }
        )
        remover = DropNearConstantColumns(max_ratio=0.5, include_na=False)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["all_null"]

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
        remover = DropNearConstantColumns(max_ratio=0.8, subset=["col1", "col2"])
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
        remover = DropNearConstantColumns(max_ratio=0.8, subset=["col1", "col2"])
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
        remover = DropNearConstantColumns(max_ratio=0.8, subset=["col2"])
        result = remover.fit_transform(X)

        # nc_outside is not in subset -> not dropped
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    # ------------------------------------------------------------------
    # fit / transform split (train / test consistency)
    # ------------------------------------------------------------------

    def test_fit_transform_separately(self):
        """Fitted transformer drops same columns on unseen data."""
        train = pl.DataFrame({"nc": [1] * 9 + [2], "varying": list(range(10))})
        test = pl.DataFrame({"nc": list(range(5)), "varying": list(range(5))})

        remover = DropNearConstantColumns(max_ratio=0.8)
        remover.fit(train)
        result = remover.transform(test)

        expected = pl.DataFrame({"varying": list(range(5))})
        assert_frame_equal(result, expected)

    def test_transform_uses_fitted_columns_not_new_data(self):
        """transform() honours fit-time decision even if column changed in test data."""
        train = pl.DataFrame({"nc": [1] * 9 + [2], "varying": list(range(10))})
        # In test data 'nc' is now fully varying
        test = pl.DataFrame({"nc": list(range(10)), "varying": list(range(10))})

        remover = DropNearConstantColumns(max_ratio=0.8)
        remover.fit(train)
        result = remover.transform(test)

        expected = pl.DataFrame({"varying": list(range(10))})
        assert_frame_equal(result, expected)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_row_dataframe(self):
        """A single row is trivially 100% one value, so it is always dropped
        regardless of max_ratio (there is no variation to observe)."""
        X = pl.DataFrame({"col1": [1], "col2": ["A"]})
        remover = DropNearConstantColumns(max_ratio=0.99)
        result = remover.fit_transform(X)
        assert result.shape[1] == 0
        assert set(remover._to_drop) == {"col1", "col2"}

    def test_empty_dataframe(self):
        """Empty DataFrame (zero rows) has no rows to evaluate dominance over,
        so no columns are dropped."""
        X = pl.DataFrame({"col1": [], "col2": []}).cast({"col1": pl.Int64, "col2": pl.String})
        remover = DropNearConstantColumns(max_ratio=0.01)
        result = remover.fit_transform(X)
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_boolean_near_constant_column(self):
        """Boolean columns are handled correctly."""
        X = pl.DataFrame(
            {
                "mostly_true": [True] * 9 + [False],
                "mixed": [True, False] * 5,
            }
        )
        # mostly_true mode share = 0.9, mixed mode share = 0.5 -> both kept at 0.95
        remover = DropNearConstantColumns(max_ratio=0.95)
        result = remover.fit_transform(X)
        assert_frame_equal(result, X)
        assert remover._to_drop == []

    def test_boolean_constant_dropped_at_high_max_ratio(self):
        """A boolean column with only one value is dropped."""
        X = pl.DataFrame(
            {
                "all_true": [True] * 10,
                "mixed": [True, False] * 5,
            }
        )
        remover = DropNearConstantColumns(max_ratio=0.8)
        result = remover.fit_transform(X)

        expected = pl.DataFrame({"mixed": [True, False] * 5})
        assert_frame_equal(result, expected)
        assert remover._to_drop == ["all_true"]

    # ------------------------------------------------------------------
    # Sklearn API compatibility
    # ------------------------------------------------------------------

    def test_get_params(self):
        """get_params() returns correct parameter dict."""
        remover = DropNearConstantColumns(max_ratio=0.05, subset=["a"], include_na=False)
        params = remover.get_params()
        assert params["max_ratio"] == 0.05
        assert params["subset"] == ["a"]
        assert params["include_na"] is False

    def test_set_params(self):
        """set_params() updates parameters correctly."""
        remover = DropNearConstantColumns()
        remover.set_params(max_ratio=0.05, include_na=False)
        assert remover.max_ratio == 0.05
        assert remover.include_na is False
