import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.feature_generation import ConditionFeatures


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pl.DataFrame(
        {
            "age": [15, 25, 30, 17, 45],
            "amount": [100, 1500, 500, 200, 2000],
            "family_size": [1, 3, 1, 4, 2],
            "velocity_24h": [1, 3, 5, 0, 10],
            "velocity_7d": [5, 8, 10, 2, 15],
        }
    )


class TestConditionFeatures:
    """Test suite for ConditionFeatures transformer."""

    def test_single_condition_scalar(self, sample_data):
        """Test with a single scalar comparison condition."""
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": ">=", "value": 18}],
            new_column_names=["is_adult"],
        )
        result = transformer.fit_transform(sample_data)

        assert "is_adult" in result.columns
        assert result["is_adult"].dtype == pl.Boolean
        assert result["is_adult"].to_list() == [False, True, True, False, True]

    def test_multiple_conditions_scalar(self, sample_data):
        """Test with multiple independent scalar comparisons."""
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">=", "value": 18},
                {"column": "amount", "op": ">", "value": 1000},
                {"column": "family_size", "op": "==", "value": 1},
            ],
            new_column_names=["is_adult", "is_high_amount", "is_alone"],
        )
        result = transformer.fit_transform(sample_data)

        assert "is_adult" in result.columns
        assert "is_high_amount" in result.columns
        assert "is_alone" in result.columns

        assert result["is_adult"].to_list() == [False, True, True, False, True]
        assert result["is_high_amount"].to_list() == [False, True, False, False, True]
        assert result["is_alone"].to_list() == [True, False, True, False, False]

    def test_column_to_column_comparison(self, sample_data):
        """Test column-to-column comparison."""
        transformer = ConditionFeatures(
            conditions=[{"column": "velocity_24h", "op": ">", "other_column": "velocity_7d"}],
            new_column_names=["is_24h_greater_than_7d"],
        )
        result = transformer.fit_transform(sample_data)

        assert "is_24h_greater_than_7d" in result.columns
        # All should be False since 24h should typically be less than 7d
        assert result["is_24h_greater_than_7d"].to_list() == [
            False,
            False,
            False,
            False,
            False,
        ]

    def test_all_comparison_operators(self, sample_data):
        """Test all supported comparison operators."""
        # Test >
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": ">", "value": 25}],
            new_column_names=["age_gt_25"],
        )
        result = transformer.fit_transform(sample_data)
        assert result["age_gt_25"].to_list() == [False, False, True, False, True]

        # Test <
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": "<", "value": 25}],
            new_column_names=["age_lt_25"],
        )
        result = transformer.fit_transform(sample_data)
        assert result["age_lt_25"].to_list() == [True, False, False, True, False]

        # Test >=
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": ">=", "value": 25}],
            new_column_names=["age_gte_25"],
        )
        result = transformer.fit_transform(sample_data)
        assert result["age_gte_25"].to_list() == [False, True, True, False, True]

        # Test <=
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": "<=", "value": 25}],
            new_column_names=["age_lte_25"],
        )
        result = transformer.fit_transform(sample_data)
        assert result["age_lte_25"].to_list() == [True, True, False, True, False]

        # Test ==
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": "==", "value": 25}],
            new_column_names=["age_eq_25"],
        )
        result = transformer.fit_transform(sample_data)
        assert result["age_eq_25"].to_list() == [False, True, False, False, False]

        # Test !=
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": "!=", "value": 25}],
            new_column_names=["age_ne_25"],
        )
        result = transformer.fit_transform(sample_data)
        assert result["age_ne_25"].to_list() == [True, False, True, True, True]

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self."""
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": ">", "value": 18}],
            new_column_names=["is_adult"],
        )
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_fit_with_y_parameter(self, sample_data):
        """Test that fit works with y parameter (sklearn compatibility)."""
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": ">", "value": 18}],
            new_column_names=["is_adult"],
        )
        y = pl.Series([0, 1, 0, 1, 0])
        result = transformer.fit(sample_data, y)
        assert result is transformer

    def test_original_columns_preserved(self, sample_data):
        """Test that original columns are preserved."""
        transformer = ConditionFeatures(
            conditions=[{"column": "age", "op": ">=", "value": 18}],
            new_column_names=["is_adult"],
        )
        result = transformer.fit_transform(sample_data)

        # Check all original columns are present
        for col in sample_data.columns:
            assert col in result.columns

        # Check original data is unchanged
        for col in sample_data.columns:
            assert result[col].to_list() == sample_data[col].to_list()

    def test_mixed_scalar_and_column_comparisons(self, sample_data):
        """Test mixing scalar and column-to-column comparisons."""
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">=", "value": 18},
                {"column": "velocity_24h", "op": "<", "other_column": "velocity_7d"},
            ],
            new_column_names=["is_adult", "is_24h_less_than_7d"],
        )
        result = transformer.fit_transform(sample_data)

        assert result["is_adult"].to_list() == [False, True, True, False, True]
        assert result["is_24h_less_than_7d"].to_list() == [
            True,
            True,
            True,
            True,
            True,
        ]


class TestConditionFeaturesValidation:
    """Test validation logic for ConditionFeatures."""

    def test_empty_conditions_raises_error(self):
        """Test that empty conditions list raises error."""
        with pytest.raises(ValueError, match="At least one condition must be specified"):
            ConditionFeatures(conditions=[], new_column_names=[])

    def test_missing_column_key_raises_error(self):
        """Test that missing 'column' key raises error."""
        with pytest.raises(ValueError, match="'column' key is required"):
            ConditionFeatures(conditions=[{"op": ">", "value": 10}], new_column_names=["test"])

    def test_missing_op_key_raises_error(self):
        """Test that missing 'op' key raises error."""
        with pytest.raises(ValueError, match="'op' key is required"):
            ConditionFeatures(
                conditions=[{"column": "age", "value": 10}], new_column_names=["test"]
            )

    def test_unsupported_operator_raises_error(self):
        """Test that unsupported operator raises error."""
        with pytest.raises(ValueError, match="operator '.*' not supported"):
            ConditionFeatures(
                conditions=[{"column": "age", "op": "between", "value": 10}],
                new_column_names=["test"],
            )

    def test_missing_value_and_other_column_raises_error(self):
        """Test that missing both value and other_column raises error."""
        with pytest.raises(ValueError, match="must specify either 'value' or 'other_column'"):
            ConditionFeatures(conditions=[{"column": "age", "op": ">"}], new_column_names=["test"])

    def test_both_value_and_other_column_raises_error(self):
        """Test that specifying both value and other_column raises error."""
        with pytest.raises(ValueError, match="cannot specify both 'value' and 'other_column'"):
            ConditionFeatures(
                conditions=[{"column": "age", "op": ">", "value": 10, "other_column": "income"}],
                new_column_names=["test"],
            )

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raises error."""
        with pytest.raises(
            ValueError,
            match="Number of column names .* must match number of conditions",
        ):
            ConditionFeatures(
                conditions=[
                    {"column": "age", "op": ">", "value": 18},
                    {"column": "amount", "op": ">", "value": 1000},
                ],
                new_column_names=["is_adult"],  # Only one name for two conditions
            )

    def test_duplicate_column_names_raises_error(self):
        """Test that duplicate column names raise error."""
        with pytest.raises(ValueError, match="Column names must be unique"):
            ConditionFeatures(
                conditions=[
                    {"column": "age", "op": ">", "value": 18},
                    {"column": "amount", "op": ">", "value": 1000},
                ],
                new_column_names=["test", "test"],  # Duplicate names
            )

    def test_empty_column_names_raises_error(self):
        """Test that empty (but not None) column names raises error."""
        with pytest.raises(ValueError, match="Column names list cannot be empty"):
            ConditionFeatures(
                conditions=[{"column": "age", "op": ">", "value": 18}],
                new_column_names=[],  # Empty list, not None
            )


class TestConditionFeaturesEdgeCases:
    """Test edge cases for ConditionFeatures."""

    def test_with_null_values(self):
        """Test behavior with null values."""
        X = pl.DataFrame({"age": [25, None, 30, None, 45], "amount": [100, 500, None, 200, 2000]})

        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">=", "value": 18},
                {"column": "amount", "op": ">", "value": 1000},
            ],
            new_column_names=["is_adult", "is_high_amount"],
        )
        result = transformer.fit_transform(X)

        # Nulls in comparisons should result in null/false
        assert result["is_adult"][0] is True
        assert result["is_adult"][1] is None
        assert result["is_high_amount"][2] is None

    def test_boolean_column_comparison(self):
        """Test comparison with boolean columns."""
        X = pl.DataFrame(
            {
                "is_active": [True, False, True, False],
                "is_verified": [True, True, False, False],
            }
        )

        transformer = ConditionFeatures(
            conditions=[{"column": "is_active", "op": "==", "value": True}],
            new_column_names=["active_check"],
        )
        result = transformer.fit_transform(X)

        assert result["active_check"].to_list() == [True, False, True, False]

    def test_string_column_comparison(self):
        """Test comparison with string columns."""
        X = pl.DataFrame({"category": ["A", "B", "A", "C", "B"]})

        transformer = ConditionFeatures(
            conditions=[
                {"column": "category", "op": "==", "value": "A"},
                {"column": "category", "op": "!=", "value": "B"},
            ],
            new_column_names=["is_category_a", "is_not_category_b"],
        )
        result = transformer.fit_transform(X)

        assert result["is_category_a"].to_list() == [True, False, True, False, False]
        assert result["is_not_category_b"].to_list() == [True, False, True, True, False]


class TestConditionFeaturesAutoNaming:
    """Test auto-naming functionality for ConditionFeatures."""

    def test_auto_naming_scalar_comparisons(self, sample_data):
        """Test auto-generated names for scalar comparisons."""
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">=", "value": 18},
                {"column": "amount", "op": ">", "value": 1000},
                {"column": "family_size", "op": "==", "value": 1},
            ]
            # No new_column_names specified
        )
        result = transformer.fit_transform(sample_data)

        # Check auto-generated column names
        assert "age_gte_18" in result.columns
        assert "amount_gt_1000" in result.columns
        assert "family_size_eq_1" in result.columns

        # Check values are correct
        assert result["age_gte_18"].to_list() == [False, True, True, False, True]
        assert result["amount_gt_1000"].to_list() == [False, True, False, False, True]
        assert result["family_size_eq_1"].to_list() == [True, False, True, False, False]

    def test_auto_naming_column_comparison(self, sample_data):
        """Test auto-generated names for column-to-column comparisons."""
        transformer = ConditionFeatures(
            conditions=[{"column": "velocity_24h", "op": "<", "other_column": "velocity_7d"}]
        )
        result = transformer.fit_transform(sample_data)

        assert "velocity_24h_lt_velocity_7d" in result.columns
        assert result["velocity_24h_lt_velocity_7d"].to_list() == [
            True,
            True,
            True,
            True,
            True,
        ]

    def test_auto_naming_all_operators(self, sample_data):
        """Test auto-naming with all supported operators."""
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">", "value": 20},
                {"column": "age", "op": "<", "value": 40},
                {"column": "age", "op": ">=", "value": 25},
                {"column": "age", "op": "<=", "value": 30},
                {"column": "age", "op": "==", "value": 25},
                {"column": "age", "op": "!=", "value": 30},
            ]
        )
        result = transformer.fit_transform(sample_data)

        assert "age_gt_20" in result.columns
        assert "age_lt_40" in result.columns
        assert "age_gte_25" in result.columns
        assert "age_lte_30" in result.columns
        assert "age_eq_25" in result.columns
        assert "age_ne_30" in result.columns

    def test_auto_naming_with_float_values(self):
        """Test auto-naming handles float values correctly."""
        X = pl.DataFrame({"score": [0.5, 1.2, 2.0, 3.5, 4.0]})

        transformer = ConditionFeatures(
            conditions=[
                {"column": "score", "op": ">", "value": 1.5},
                {
                    "column": "score",
                    "op": ">=",
                    "value": 2.0,
                },  # Should format as 2, not 2.0
            ]
        )
        result = transformer.fit_transform(X)

        assert "score_gt_1.5" in result.columns
        assert "score_gte_2" in result.columns  # No .0 for integer-valued float

    def test_auto_naming_mixed_with_column_comparison(self, sample_data):
        """Test auto-naming with mix of scalar and column comparisons."""
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">=", "value": 18},
                {"column": "velocity_24h", "op": ">", "other_column": "velocity_7d"},
            ]
        )
        result = transformer.fit_transform(sample_data)

        assert "age_gte_18" in result.columns
        assert "velocity_24h_gt_velocity_7d" in result.columns

    def test_is_null_operator(self):
        """Test is_null unary operator."""
        X = pl.DataFrame({"age": [25, None, 30, None, 45], "amount": [100, 500, None, 200, 2000]})

        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": "is_null"},
                {"column": "amount", "op": "is_null"},
            ],
            new_column_names=["age_missing", "amount_missing"],
        )
        result = transformer.fit_transform(X)

        assert "age_missing" in result.columns
        assert "amount_missing" in result.columns
        assert result["age_missing"].to_list() == [False, True, False, True, False]
        assert result["amount_missing"].to_list() == [False, False, True, False, False]

    def test_is_not_null_operator(self):
        """Test is_not_null unary operator."""
        X = pl.DataFrame(
            {
                "email": ["a@test.com", None, "c@test.com", None, "e@test.com"],
                "phone": [None, "123", None, "456", "789"],
            }
        )

        transformer = ConditionFeatures(
            conditions=[
                {"column": "email", "op": "is_not_null"},
                {"column": "phone", "op": "is_not_null"},
            ],
            new_column_names=["has_email", "has_phone"],
        )
        result = transformer.fit_transform(X)

        assert "has_email" in result.columns
        assert "has_phone" in result.columns
        assert result["has_email"].to_list() == [True, False, True, False, True]
        assert result["has_phone"].to_list() == [False, True, False, True, True]

    def test_null_operators_auto_naming(self):
        """Test auto-naming for null check operators."""
        X = pl.DataFrame({"age": [25, None, 30], "email": ["a@test.com", None, "c@test.com"]})

        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": "is_null"},
                {"column": "email", "op": "is_not_null"},
            ]
        )
        result = transformer.fit_transform(X)

        assert "age__is_null" in result.columns
        assert "email__is_not_null" in result.columns
        assert result["age__is_null"].to_list() == [False, True, False]
        assert result["email__is_not_null"].to_list() == [True, False, True]

    def test_mixed_conditions_with_null_checks(self):
        """Test mixing null checks with regular comparisons."""
        X = pl.DataFrame(
            {
                "age": [15, 25, None, 17, 45],
                "amount": [100, 1500, 500, None, 2000],
            }
        )

        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">=", "value": 18},
                {"column": "age", "op": "is_null"},
                {"column": "amount", "op": ">", "value": 1000},
                {"column": "amount", "op": "is_not_null"},
            ],
            new_column_names=[
                "is_adult",
                "age_missing",
                "is_high_amount",
                "has_amount",
            ],
        )
        result = transformer.fit_transform(X)

        assert result["is_adult"].to_list() == [False, True, None, False, True]
        assert result["age_missing"].to_list() == [False, False, True, False, False]
        assert result["is_high_amount"].to_list() == [False, True, False, None, True]
        assert result["has_amount"].to_list() == [True, True, True, False, True]

    def test_null_operator_validation_error_with_value(self):
        """Test that is_null operator rejects 'value' parameter."""
        with pytest.raises(
            ValueError,
            match="unary operator 'is_null' should not have 'value' or 'other_column'",
        ):
            ConditionFeatures(conditions=[{"column": "age", "op": "is_null", "value": 18}])

    def test_null_operator_validation_error_with_other_column(self):
        """Test that is_not_null operator rejects 'other_column' parameter."""
        with pytest.raises(
            ValueError,
            match="unary operator 'is_not_null' should not have 'value' or 'other_column'",
        ):
            ConditionFeatures(
                conditions=[{"column": "age", "op": "is_not_null", "other_column": "amount"}]
            )


class TestStaticMethods:
    """Test coverage for static methods with invalid operators."""

    def test_build_scalar_comparison_invalid_operator(self):
        """Test _build_scalar_comparison with an invalid operator."""
        with pytest.raises(ValueError, match="Unsupported operator: invalid_op"):
            ConditionFeatures._build_scalar_comparison("age", "invalid_op", 18)

    def test_build_column_comparison_invalid_operator(self):
        """Test _build_column_comparison with an invalid operator."""
        with pytest.raises(ValueError, match="Unsupported operator: invalid_op"):
            ConditionFeatures._build_column_comparison("age", "invalid_op", "amount")

    def test_build_unary_operation_invalid_operator(self):
        """Test _build_unary_operation with an invalid operator."""
        with pytest.raises(ValueError, match="Unsupported unary operator: invalid_op"):
            ConditionFeatures._build_unary_operation("age", "invalid_op")

    def test_column_comparison_all_operators(self):
        """Test column-to-column comparison with all operators to ensure coverage."""
        sample_data = pl.DataFrame(
            {
                "age": [15, 25, 30, 17, 45],
                "amount": [100, 1500, 500, 200, 2000],
            }
        )

        # Test all operators: <, >=, <=, ==, != (> is already tested elsewhere)
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": "<", "other_column": "amount"},
                {"column": "age", "op": ">=", "other_column": "amount"},
                {"column": "age", "op": "<=", "other_column": "amount"},
                {"column": "age", "op": "==", "other_column": "amount"},
                {"column": "age", "op": "!=", "other_column": "amount"},
            ],
            new_column_names=[
                "age_lt_amount",
                "age_gte_amount",
                "age_lte_amount",
                "age_eq_amount",
                "age_neq_amount",
            ],
        )
        result = transformer.fit_transform(sample_data)

        # Verify all columns were created
        assert "age_lt_amount" in result.columns
        assert "age_gte_amount" in result.columns
        assert "age_lte_amount" in result.columns
        assert "age_eq_amount" in result.columns
        assert "age_neq_amount" in result.columns

    def test_empty_new_column_names_list(self):
        """Test that empty list for new_column_names raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Column names list cannot be empty",
        ):
            ConditionFeatures(
                conditions=[{"column": "age", "op": ">", "value": 18}],
                new_column_names=[],  # Empty list should raise error
            )

    def test_none_new_column_names(self):
        """Test that None for new_column_names triggers auto-naming."""
        sample_data = pl.DataFrame(
            {
                "age": [15, 25, 30],
            }
        )

        # Using None should trigger auto-naming
        transformer = ConditionFeatures(
            conditions=[
                {"column": "age", "op": ">", "value": 18},
                {"column": "age", "op": "<", "value": 65},
            ],
            new_column_names=None,  # Auto-naming
        )
        result = transformer.fit_transform(sample_data)

        # verify auto-generated column names exist
        assert len(result.columns) == 3  # Original 1 + 2 new ones


if __name__ == "__main__":
    pytest.main([__file__])
