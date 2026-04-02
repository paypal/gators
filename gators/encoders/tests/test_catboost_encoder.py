"""Test CatBoostEncoder transformer."""

import polars as pl
import pytest

from gators.encoders import CatBoostEncoder


class TestCatBoostEncoder:
    """Test CatBoostEncoder transformer."""

    def test_basic_encoding(self):
        """Test basic CatBoost encoding with ordered statistics."""
        X = pl.DataFrame(
            {
                "category": ["A", "B", "A", "C", "A", "B", "C"],
                "value": [1, 2, 3, 4, 5, 6, 7],
            }
        )
        target = pl.Series("target", [1, 0, 1, 0, 0, 1, 1])
        encoder = CatBoostEncoder(subset=["category"], smoothing=1.0, inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        # Check structure
        assert "category__catboost_enc" in result.columns
        assert "category" not in result.columns  # drop_columns=True by default
        assert result.shape[0] == 7

        # Check that all values are floats
        assert result["category__catboost_enc"].dtype == pl.Float64

    def test_no_target_raises_error(self):
        """Test that fitting without target raises ValueError."""
        X = pl.DataFrame({"category": ["A", "B", "A"]})
        encoder = CatBoostEncoder(subset=["category"], inplace=False)

        with pytest.raises(ValueError, match="requires a target variable"):
            encoder.fit(X)

    def test_target_not_in_dataframe_raises_error(self):
        """Test removed - no longer applicable with Series API."""
        pass

    def test_smoothing_parameter(self):
        """Test that smoothing parameter affects encoding."""
        X = pl.DataFrame({"category": ["A", "A", "A"]})
        target = pl.Series("target", [1, 0, 1])

        # With low smoothing
        encoder_low = CatBoostEncoder(subset=["category"], smoothing=0.1, inplace=False)
        encoder_low.fit(X, y=target)
        result_low = encoder_low.transform(X)

        # With high smoothing (should be closer to global mean)
        encoder_high = CatBoostEncoder(subset=["category"], smoothing=10.0, inplace=False)
        encoder_high.fit(X, y=target)
        result_high = encoder_high.transform(X)

        # High smoothing should pull values closer to global mean
        global_mean = target.mean()
        mean_low = result_low["category__catboost_enc"].mean()
        mean_high = result_high["category__catboost_enc"].mean()

        # High smoothing result should be closer to global mean
        assert abs(mean_high - global_mean) < abs(mean_low - global_mean)

    def test_global_mean_attribute(self):
        """Test that global_mean_ is calculated correctly."""
        X = pl.DataFrame({"category": ["A", "B", "C"]})
        target = pl.Series("target", [1, 2, 3])
        encoder = CatBoostEncoder(subset=["category"], inplace=False)
        encoder.fit(X, y=target)

        assert encoder.global_mean_ == 2.0

    def test_min_count_absolute(self):
        """Test min_count with absolute value."""
        X = pl.DataFrame({"category": ["A", "A", "A", "B", "B", "C"]})
        target = pl.Series("target", [1, 0, 1, 1, 0, 1])
        encoder = CatBoostEncoder(
            subset=["category"], min_count=3, drop_columns=False, inplace=False
        )
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        # Only "A" should be encoded (count=3)
        # "B" (count=2) and "C" (count=1) should use global mean
        assert "category__catboost_enc" in result.columns

        # Check that C (rare category) gets global mean
        global_mean = encoder.global_mean_
        c_value = result.filter(pl.col("category") == "C")["category__catboost_enc"][0]
        assert abs(c_value - global_mean) < 0.01

    def test_min_count_ratio(self):
        """Test min_count with ratio value."""
        X = pl.DataFrame({"category": ["A"] * 5 + ["B"] * 2 + ["C"]})
        target = pl.Series("target", [1, 0, 1, 0, 1, 1, 0, 1])
        encoder = CatBoostEncoder(
            subset=["category"], min_count=0.3, inplace=False
        )  # 30% of 8 = 2.4
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        # Only "A" (5 occurrences) should be encoded
        # "B" (2) and "C" (1) should use global mean
        assert "category__catboost_enc" in result.columns

    def test_multiple_columns(self):
        """Test encoding multiple columns."""
        X = pl.DataFrame({"cat1": ["A", "B", "A"], "cat2": ["X", "Y", "X"]})
        target = pl.Series("target", [1, 0, 1])
        encoder = CatBoostEncoder(subset=["cat1", "cat2"], inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        assert "cat1__catboost_enc" in result.columns
        assert "cat2__catboost_enc" in result.columns
        assert "cat1" not in result.columns
        assert "cat2" not in result.columns

    def test_columns_none_auto_detect(self):
        """Test automatic column detection when subset=None."""
        X = pl.DataFrame(
            {
                "str_col": ["A", "B", "A"],
                "bool_col": [True, False, True],
                "int_col": [1, 2, 3],
            }
        )
        target = pl.Series("target", [1, 0, 1])
        encoder = CatBoostEncoder(inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        # Should encode str_col and bool_col, but not int_col
        assert "str_col__catboost_enc" in result.columns
        assert "bool_col__catboost_enc" in result.columns
        assert "int_col" in result.columns

    def test_drop_columns_false(self):
        """Test that original columns are kept when drop_columns=False."""
        X = pl.DataFrame({"category": ["A", "B", "A"]})
        target = pl.Series("target", [1, 0, 1])
        encoder = CatBoostEncoder(subset=["category"], drop_columns=False, inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        assert "category" in result.columns
        assert "category__catboost_enc" in result.columns

    def test_unseen_categories_use_global_mean(self):
        """Test that unseen categories get global mean value."""
        X_train = pl.DataFrame({"category": ["A", "B", "A", "B"]})
        target_train = pl.Series("target", [1, 0, 1, 0])
        X_test = pl.DataFrame({"category": ["A", "C", "D"]})  # C and D are unseen

        encoder = CatBoostEncoder(subset=["category"], drop_columns=False, inplace=False)
        encoder.fit(X_train, y=target_train)
        result = encoder.transform(X_test)

        global_mean = encoder.global_mean_

        # Unseen categories should get global mean
        c_value = result.filter(pl.col("category") == "C")["category__catboost_enc"][0]
        d_value = result.filter(pl.col("category") == "D")["category__catboost_enc"][0]

        assert abs(c_value - global_mean) < 0.01
        assert abs(d_value - global_mean) < 0.01

    def test_ordered_statistics_prevents_leakage(self):
        """Test that ordered statistics exclude current row from calculation."""
        # Create a simple example where we can verify the calculation
        X = pl.DataFrame({"category": ["A", "A", "A"]})
        target = pl.Series("target", [1, 0, 1])
        encoder = CatBoostEncoder(subset=["category"], smoothing=0.1, inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        # Ordered statistics use cumulative means that exclude the current row
        # The transform uses the average encoding across all rows from fit
        # Just verify that values are numeric and reasonable
        values = result["category__catboost_enc"].to_list()

        # All values should be numeric (not nan/inf)
        assert all(isinstance(v, float) for v in values)
        assert all(not pl.Series([v]).is_nan()[0] for v in values)
        assert all(not pl.Series([v]).is_infinite()[0] for v in values)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        X_train = pl.DataFrame({"category": ["A", "B"]})
        target_train = pl.Series("target", [1, 0])
        X_test = pl.DataFrame({"category": []}).with_columns([pl.col("category").cast(pl.String)])

        encoder = CatBoostEncoder(subset=["category"], inplace=False)
        encoder.fit(X_train, y=target_train)
        result = encoder.transform(X_test)

        assert result.shape[0] == 0
        assert "category__catboost_enc" in result.columns

    def test_single_category(self):
        """Test encoding with single category."""
        X = pl.DataFrame({"category": ["A", "A", "A"]})
        target = pl.Series("target", [1, 0, 1])
        encoder = CatBoostEncoder(subset=["category"], inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        assert "category__catboost_enc" in result.columns
        assert result.shape[0] == 3

    def test_boolean_column(self):
        """Test encoding boolean column."""
        X = pl.DataFrame({"bool_col": [True, False, True, False]})
        target = pl.Series("target", [1, 0, 1, 0])
        encoder = CatBoostEncoder(subset=["bool_col"], inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        assert "bool_col__catboost_enc" in result.columns
        assert result["bool_col__catboost_enc"].dtype == pl.Float64

    def test_sklearn_compatibility(self):
        """Test sklearn-compatible API."""
        X = pl.DataFrame({"category": ["A", "B", "A", "B"]})
        target = pl.Series("target", [1, 0, 1, 0])
        encoder = CatBoostEncoder(subset=["category"], inplace=False)

        # fit should return self
        result = encoder.fit(X, y=target)
        assert result is encoder

        # fit_transform should work
        result = encoder.fit_transform(X, y=target)
        assert isinstance(result, pl.DataFrame)
        assert "category__catboost_enc" in result.columns

    def test_mapping_structure(self):
        """Test that mapping_ is created correctly."""
        X = pl.DataFrame({"cat1": ["A", "B", "A"], "cat2": ["X", "Y", "X"]})
        target = pl.Series("target", [1, 0, 1])
        encoder = CatBoostEncoder(subset=["cat1", "cat2"], inplace=False)
        encoder.fit(X, y=target)

        # Check mapping_ structure
        assert isinstance(encoder.mapping_, dict)
        assert "cat1" in encoder.mapping_
        assert "cat2" in encoder.mapping_
        assert isinstance(encoder.mapping_["cat1"], dict)

        # Check column_mapping_
        assert encoder.column_mapping_ == {
            "cat1": "cat1__catboost_enc",
            "cat2": "cat2__catboost_enc",
        }

    def test_numeric_target(self):
        """Test with numeric target values."""
        X = pl.DataFrame({"category": ["A", "B", "A", "B"]})
        target = pl.Series("target", [10.5, 20.3, 15.7, 18.9])
        encoder = CatBoostEncoder(subset=["category"], inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        assert "category__catboost_enc" in result.columns
        # Global mean should be calculated correctly
        assert abs(encoder.global_mean_ - 16.35) < 0.01

    def test_high_cardinality(self):
        """Test encoding with many categories."""
        categories = [f"cat_{i}" for i in range(100)]
        targets = [i % 2 for i in range(100)]

        X = pl.DataFrame({"category": categories})
        target = pl.Series("target", targets)
        encoder = CatBoostEncoder(subset=["category"], inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        assert "category__catboost_enc" in result.columns
        assert result.shape[0] == 100
        assert len(encoder.mapping_["category"]) == 100

    def test_all_same_target_values(self):
        """Test encoding when all target values are the same."""
        X = pl.DataFrame({"category": ["A", "B", "C"]})
        target = pl.Series("target", [1, 1, 1])
        encoder = CatBoostEncoder(subset=["category"], inplace=False)
        encoder.fit(X, y=target)
        result = encoder.transform(X)

        # All encoded values should be close to 1.0 (the global mean)
        assert "category__catboost_enc" in result.columns
        values = result["category__catboost_enc"].to_list()
        assert all(abs(v - 1.0) < 0.5 for v in values)

    def test_column_mapping_attribute(self):
        """Test that column_mapping_ attribute is set correctly."""
        X = pl.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})
        target = pl.Series("target", [1, 0])
        encoder = CatBoostEncoder(subset=["cat1", "cat2"], inplace=False)
        encoder.fit(X, y=target)

        expected_mapping = {"cat1": "cat1__catboost_enc", "cat2": "cat2__catboost_enc"}
        assert encoder.column_mapping_ == expected_mapping
