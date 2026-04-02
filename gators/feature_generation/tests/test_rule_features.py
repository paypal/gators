import polars as pl
import pytest

from gators.feature_generation import RuleFeatures


class TestRuleFeatures:
    """Test suite for RuleFeatures transformer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "amount": [100, 500, 1200, 50, 2000],
                "velocity_24h": [1, 3, 5, 0, 10],
                "velocity_7d": [5, 8, 10, 2, 15],
                "is_new_user": [True, False, False, True, False],
                "age": [25, 35, 45, 20, 55],
            }
        )

    def test_single_rule_and_logic(self, sample_data):
        """Test single rule with AND logic and scalar value comparisons."""
        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "amount", "op": ">", "value": 1000},
                    {"column": "velocity_24h", "op": ">=", "value": 5},
                ]
            ],
            rule_logic="and",
            new_column_names=["high_risk"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "high_risk" in result.columns
        assert result["high_risk"].dtype == pl.Boolean
        # Row 2: amount=1200 (>1000) AND velocity_24h=5 (>=5) -> True
        # Row 4: amount=2000 (>1000) AND velocity_24h=10 (>=5) -> True
        assert result["high_risk"].to_list() == [False, False, True, False, True]

    def test_single_rule_or_logic(self, sample_data):
        """Test single rule with OR logic and scalar value comparisons."""
        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "amount", "op": ">", "value": 1000},
                    {"column": "velocity_24h", "op": ">=", "value": 5},
                ]
            ],
            rule_logic="or",
            new_column_names=["risk_flag"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "risk_flag" in result.columns
        # Row 2: amount=1200 (>1000) OR velocity_24h=5 (>=5) -> True
        # Row 4: amount=2000 (>1000) OR velocity_24h=10 (>=5) -> True
        assert result["risk_flag"].to_list() == [False, False, True, False, True]

    def test_multiple_rules(self, sample_data):
        """Test creating multiple rule outputs in one pass."""
        transformer = RuleFeatures(
            rules=[
                # Rule 1: High amount
                [{"column": "amount", "op": ">", "value": 1000}],
                # Rule 2: High velocity
                [{"column": "velocity_24h", "op": ">=", "value": 5}],
            ],
            rule_logic="and",
            new_column_names=["is_high_amount", "is_high_velocity"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "is_high_amount" in result.columns
        assert "is_high_velocity" in result.columns
        # High amount: rows 2 and 4 (1200, 2000)
        assert result["is_high_amount"].to_list() == [False, False, True, False, True]
        # High velocity: rows 2 and 4 (5, 10)
        assert result["is_high_velocity"].to_list() == [False, False, True, False, True]

    def test_complex_multiple_rules(self, sample_data):
        """Test multiple rules with different condition counts."""
        transformer = RuleFeatures(
            rules=[
                # Rule 1: New user AND high amount (2 conditions)
                [
                    {"column": "is_new_user", "op": "==", "value": True},
                    {"column": "amount", "op": ">", "value": 1000},
                ],
                # Rule 2: High velocity (1 condition)
                [{"column": "velocity_24h", "op": ">=", "value": 10}],
                # Rule 3: Young AND high spender (2 conditions)
                [
                    {"column": "age", "op": "<", "value": 30},
                    {"column": "amount", "op": ">", "value": 500},
                ],
            ],
            rule_logic="and",
            new_column_names=[
                "suspicious_new",
                "extreme_velocity",
                "young_big_spender",
            ],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "suspicious_new" in result.columns
        assert "extreme_velocity" in result.columns
        assert "young_big_spender" in result.columns
        # suspicious_new: No new users with amount > 1000
        assert result["suspicious_new"].to_list() == [False, False, False, False, False]
        # extreme_velocity: Only row 4 has velocity_24h >= 10
        assert result["extreme_velocity"].to_list() == [
            False,
            False,
            False,
            False,
            True,
        ]
        # young_big_spender: Row 0 (age=25, amount=100 - not > 500), Row 3 (age=20, amount=50 - not > 500)
        assert result["young_big_spender"].to_list() == [
            False,
            False,
            False,
            False,
            False,
        ]

    def test_column_to_column_comparison(self, sample_data):
        """Test comparison between two columns."""
        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "velocity_24h", "op": ">", "value": 0},
                    {
                        "column": "velocity_7d",
                        "op": "==",
                        "other_column": "velocity_24h",
                    },
                ]
            ],
            rule_logic="and",
            new_column_names=["is_spike"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "is_spike" in result.columns
        # No rows have velocity_7d == velocity_24h
        assert result["is_spike"].to_list() == [False, False, False, False, False]

    def test_drop_conditions_false(self, sample_data):
        """Test that intermediate condition columns are kept when drop_conditions=False."""
        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "amount", "op": ">", "value": 500},
                    {"column": "age", "op": "<", "value": 30},
                ]
            ],
            rule_logic="and",
            new_column_names=["young_high_spender"],
            drop_conditions=False,
        )

        result = transformer.fit_transform(sample_data)

        # Should have intermediate condition columns
        assert "_rule_0_cond_0_young_high_spender" in result.columns
        assert "_rule_0_cond_1_young_high_spender" in result.columns
        assert "young_high_spender" in result.columns

    def test_drop_conditions_true(self, sample_data):
        """Test that intermediate condition columns are removed when drop_conditions=True."""
        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "amount", "op": ">", "value": 500},
                    {"column": "age", "op": "<", "value": 30},
                ]
            ],
            rule_logic="and",
            new_column_names=["young_high_spender"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        # Should NOT have intermediate condition columns
        assert "_rule_0_cond_0_young_high_spender" not in result.columns
        assert "_rule_0_cond_1_young_high_spender" not in result.columns
        assert "young_high_spender" in result.columns

    def test_boolean_column_comparison(self, sample_data):
        """Test comparison with boolean column."""
        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "is_new_user", "op": "==", "value": True},
                    {"column": "amount", "op": ">", "value": 1000},
                ]
            ],
            rule_logic="and",
            new_column_names=["suspicious_new_user"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "suspicious_new_user" in result.columns
        # Row 0: is_new_user=True AND amount=100 (not >1000) -> False
        # Row 3: is_new_user=True AND amount=50 (not >1000) -> False
        assert result["suspicious_new_user"].to_list() == [
            False,
            False,
            False,
            False,
            False,
        ]

    def test_all_operators(self, sample_data):
        """Test all supported operators."""
        # Test >
        t1 = RuleFeatures(
            rules=[[{"column": "amount", "op": ">", "value": 100}]],
            rule_logic="and",
            new_column_names=["gt_test"],
        )
        r1 = t1.fit_transform(sample_data)
        assert r1["gt_test"].sum() == 3  # 500, 1200, 2000 > 100

        # Test <
        t2 = RuleFeatures(
            rules=[[{"column": "amount", "op": "<", "value": 100}]],
            rule_logic="and",
            new_column_names=["lt_test"],
        )
        r2 = t2.fit_transform(sample_data)
        assert r2["lt_test"].sum() == 1  # 50 < 100

        # Test >=
        t3 = RuleFeatures(
            rules=[[{"column": "amount", "op": ">=", "value": 100}]],
            rule_logic="and",
            new_column_names=["gte_test"],
        )
        r3 = t3.fit_transform(sample_data)
        assert r3["gte_test"].sum() == 4  # 100, 500, 1200, 2000 >= 100

        # Test <=
        t4 = RuleFeatures(
            rules=[[{"column": "amount", "op": "<=", "value": 100}]],
            rule_logic="and",
            new_column_names=["lte_test"],
        )
        r4 = t4.fit_transform(sample_data)
        assert r4["lte_test"].sum() == 2  # 100, 50 <= 100

        # Test ==
        t5 = RuleFeatures(
            rules=[[{"column": "amount", "op": "==", "value": 100}]],
            rule_logic="and",
            new_column_names=["eq_test"],
        )
        r5 = t5.fit_transform(sample_data)
        assert r5["eq_test"].sum() == 1  # Only one 100

        # Test !=
        t6 = RuleFeatures(
            rules=[[{"column": "amount", "op": "!=", "value": 100}]],
            rule_logic="and",
            new_column_names=["neq_test"],
        )
        r6 = t6.fit_transform(sample_data)
        assert r6["neq_test"].sum() == 4  # 500, 1200, 50, 2000 != 100

    def test_sklearn_compatibility(self, sample_data):
        """Test that transformer works in sklearn pipeline."""
        from sklearn.pipeline import Pipeline

        transformer = RuleFeatures(
            rules=[
                [
                    {"column": "amount", "op": ">", "value": 500},
                    {"column": "velocity_24h", "op": ">", "value": 0},
                ]
            ],
            rule_logic="and",
            new_column_names=["pipeline_test"],
            drop_conditions=True,
        )

        pipeline = Pipeline([("logical_features", transformer)])

        result = pipeline.fit_transform(sample_data)
        assert "pipeline_test" in result.columns

    def test_validation_empty_rules(self):
        """Test that empty rules list raises ValueError."""
        with pytest.raises(ValueError, match="At least one rule must be specified"):
            RuleFeatures(rules=[], rule_logic="and", new_column_names=[])

    def test_validation_empty_rule(self):
        """Test that empty rule raises ValueError."""
        with pytest.raises(ValueError, match="must contain at least one condition"):
            RuleFeatures(rules=[[]], rule_logic="and", new_column_names=["test"])

    def test_validation_missing_column(self):
        """Test that condition without column key raises ValueError."""
        with pytest.raises(ValueError, match="'column' key is required"):
            RuleFeatures(
                rules=[[{"op": ">", "value": 100}]],
                rule_logic="and",
                new_column_names=["test"],
            )

    def test_validation_missing_op(self):
        """Test that condition without op key raises ValueError."""
        with pytest.raises(ValueError, match="'op' key is required"):
            RuleFeatures(
                rules=[[{"column": "amount", "value": 100}]],
                rule_logic="and",
                new_column_names=["test"],
            )

    def test_validation_unsupported_operator(self):
        """Test that unsupported operator raises ValueError."""
        with pytest.raises(ValueError, match="operator '~' not supported"):
            RuleFeatures(
                rules=[[{"column": "amount", "op": "~", "value": 100}]],
                rule_logic="and",
                new_column_names=["test"],
            )

    def test_validation_missing_value_and_other_column(self):
        """Test that condition without value or other_column raises ValueError."""
        with pytest.raises(
            ValueError, match="must specify either 'value' or 'other_column'"
        ):
            RuleFeatures(
                rules=[[{"column": "amount", "op": ">"}]],
                rule_logic="and",
                new_column_names=["test"],
            )

    def test_validation_both_value_and_other_column(self):
        """Test that condition with both value and other_column raises ValueError."""
        with pytest.raises(
            ValueError, match="cannot specify both 'value' and 'other_column'"
        ):
            RuleFeatures(
                rules=[
                    [
                        {
                            "column": "amount",
                            "op": ">",
                            "value": 100,
                            "other_column": "age",
                        }
                    ]
                ],
                rule_logic="and",
                new_column_names=["test"],
            )

    def test_validation_mismatched_column_names_length(self):
        """Test that mismatched rules and column names length raises ValueError."""
        with pytest.raises(
            ValueError, match="Number of column names .* must match number of rules"
        ):
            RuleFeatures(
                rules=[
                    [{"column": "amount", "op": ">", "value": 100}],
                    [{"column": "age", "op": "<", "value": 30}],
                ],
                rule_logic="and",
                new_column_names=["only_one_name"],
            )

    def test_validation_duplicate_column_names(self):
        """Test that duplicate column names raise ValueError."""
        with pytest.raises(ValueError, match="Column names must be unique"):
            RuleFeatures(
                rules=[
                    [{"column": "amount", "op": ">", "value": 100}],
                    [{"column": "age", "op": "<", "value": 30}],
                ],
                rule_logic="and",
                new_column_names=["test", "test"],
            )

    def test_validation_empty_column_names(self):
        """Test that empty column names list raises ValueError."""
        with pytest.raises(
            ValueError, match="At least one column name must be specified"
        ):
            RuleFeatures(
                rules=[[{"column": "amount", "op": ">", "value": 100}]],
                rule_logic="and",
                new_column_names=[],
            )

    def test_single_condition_rule(self, sample_data):
        """Test rule with single condition (no combining needed)."""
        transformer = RuleFeatures(
            rules=[[{"column": "amount", "op": ">", "value": 1000}]],
            rule_logic="and",
            new_column_names=["high_amount"],
            drop_conditions=True,
        )

        result = transformer.fit_transform(sample_data)

        assert "high_amount" in result.columns
        assert result["high_amount"].to_list() == [False, False, True, False, True]

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for chaining."""
        transformer = RuleFeatures(
            rules=[[{"column": "amount", "op": ">", "value": 100}]],
            rule_logic="and",
            new_column_names=["test"],
        )

        fitted = transformer.fit(sample_data)
        assert fitted is transformer


class TestStaticMethods:
    """Test coverage for static methods with invalid operators."""

    def test_build_scalar_comparison_invalid_operator(self):
        """Test _build_scalar_comparison with an invalid operator."""
        with pytest.raises(ValueError, match="Unsupported operator: invalid_op"):
            RuleFeatures._build_scalar_comparison("amount", "invalid_op", 100)

    def test_build_column_comparison_invalid_operator(self):
        """Test _build_column_comparison with an invalid operator."""
        with pytest.raises(ValueError, match="Unsupported operator: invalid_op"):
            RuleFeatures._build_column_comparison("amount", "invalid_op", "age")

    def test_column_comparison_all_operators(self):
        """Test column-to-column comparison with all operators."""
        # Create test data
        sample_data = pl.DataFrame(
            {
                "age": [15, 25, 30, 17, 45],
                "amount": [100, 1500, 500, 200, 2000],
            }
        )
        
        # Test all operators: >, <, >=, <=, ==, !=
        transformer = RuleFeatures(
            rules=[
                [{"column": "age", "op": ">", "other_column": "amount"}],
                [{"column": "age", "op": "<", "other_column": "amount"}],
                [{"column": "age", "op": ">=", "other_column": "amount"}],
                [{"column": "age", "op": "<=", "other_column": "amount"}],
                [{"column": "age", "op": "==", "other_column": "amount"}],
                [{"column": "age", "op": "!=", "other_column": "amount"}],
            ],
            rule_logic="and",
            new_column_names=["age_gt_amount", "age_lt_amount", "age_gte_amount", "age_lte_amount", "age_eq_amount", "age_neq_amount"],
        )
        result = transformer.fit_transform(sample_data)
        
        # Verify all columns were created
        assert "age_gt_amount" in result.columns
        assert "age_lt_amount" in result.columns
        assert "age_gte_amount" in result.columns
        assert "age_lte_amount" in result.columns
        assert "age_eq_amount" in result.columns
        assert "age_neq_amount" in result.columns

    def test_less_than_operator_explicit(self):
        """Explicitly test the < operator to ensure line 360 is covered."""
        sample_data = pl.DataFrame(
            {
                "age": [15, 25, 30],
                "limit": [20, 20, 20],
            }
        )
        
        # Directly test < operator
        transformer = RuleFeatures(
            rules=[[{"column": "age", "op": "<", "other_column": "limit"}]],
            rule_logic="and",
            new_column_names=["age_below_limit"],
        )
        result = transformer.fit_transform(sample_data)
        
        assert "age_below_limit" in result.columns
        # First row: 15 < 20 = True (1)
        assert result["age_below_limit"][0] == 1
        # Second row: 25 < 20 = False (0)
        assert result["age_below_limit"][1] == 0


if __name__ == "__main__":
    pytest.main([__file__])
