from typing import Any, Dict, List, Literal, Optional

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class RuleFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Creates multiple boolean features, each from a group of conditions combined with logical operators.

    This transformer is useful for creating multiple rule-based features simultaneously,
    where each rule represents a distinct business logic or fraud detection pattern.
    Each rule group produces its own boolean output column.

    **Use Cases:**
    
    - Fraud detection: Create multiple risk indicators (velocity spike, amount anomaly, etc.)
    - Business rules: Generate several eligibility/qualification flags at once
    - Feature engineering: Build a family of related boolean features
    - Production pipelines: Encapsulate multiple rule definitions in one transformer

    **When to Use:**
    
    - Building production ML pipelines that need serialization
    - Creating reusable feature engineering templates
    - Working with sklearn-based systems that expect transformers
    - Need version control of feature logic (can serialize to JSON/YAML)
    - Want to create multiple related boolean features efficiently

    **When NOT to Use:**
    
    - One-off exploratory analysis (use Polars native expressions)
    - Very complex nested logic within a single rule (consider Polars native)
    - Performance-critical scenarios where every microsecond counts

    Parameters
    ----------
    rules : List[List[Dict[str, Any]]]
        List of rule groups. Each rule group contains condition dictionaries that will
        be combined to create one boolean output column.

        Each condition dictionary must contain:
        
        - 'column': str - Name of the column to evaluate
        - 'op': str - Comparison operator. Supported: '>', '<', '>=', '<=', '==', '!='
        - 'value': Any (optional) - Scalar value to compare the column against
        - 'other_column': str (optional) - Name of another column to compare against

        Either 'value' or 'other_column' must be specified, but not both.

        Examples::
        
            # Two rules:
            [
                [{'column': 'age', 'op': '>', 'value': 18}],
                [{'column': 'amount', 'op': '>', 'value': 1000}]
            ]
            
            # Rule with multiple conditions:
            [
                [{'column': 'age', 'op': '>', 'value': 18},
                 {'column': 'amount', 'op': '>', 'value': 1000}]
            ]

    rule_logic : Literal['and', 'or'], default='and'
        How to combine conditions within each rule group:
        
        - 'and': All conditions in a group must be True
        - 'or': At least one condition in a group must be True

    new_column_names : List[str]
        Names for the resulting boolean feature columns. Must have the same length as `rules`.
        Each rule group will produce a column with the corresponding name.

    drop_conditions : bool, default=False
        Whether to drop intermediate condition columns after combining.
        Recommended: True for cleaner output.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation import RuleFeatures

    >>> X ={
    ...     'amount': [100, 500, 1200, 50, 2000],
    ...     'velocity_24h': [1, 3, 5, 0, 10],
    ...     'velocity_7d': [5, 8, 10, 2, 15],
    ...     'is_new_user': [True, False, False, True, False]
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Create two risk indicators in one pass**

    >>> multi_risk_transformer = RuleFeatures(
    ...     rules=[
    ...         # Rule 1: Activity spike (24h > 0 AND 7d == 24h)
    ...         [
    ...             {'column': 'velocity_24h', 'op': '>', 'value': 0},
    ...             {'column': 'velocity_7d', 'op': '==', 'other_column': 'velocity_24h'}
    ...         ],
    ...         # Rule 2: High amount (amount > 1000)
    ...         [
    ...             {'column': 'amount', 'op': '>', 'value': 1000}
    ...         ]
    ...     ],
    ...     rule_logic='and',
    ...     new_column_names=['is_activity_spike', 'is_high_amount'],
    ...     drop_conditions=True
    ... )
    >>> result = multi_risk_transformer.fit_transform(X)
    >>> result.select(['velocity_24h', 'velocity_7d', 'amount',
    ...                'is_activity_spike', 'is_high_amount'])
    shape: (5, 5)
    ┌──────────────┬─────────────┬────────┬────────────────────┬─────────────────┐
    │ velocity_24h ┆ velocity_7d ┆ amount ┆ is_activity_spike  ┆ is_high_amount  │
    │ ---          ┆ ---         ┆ ---    ┆ ---                ┆ ---             │
    │ i64          ┆ i64         ┆ i64    ┆ bool               ┆ bool            │
    ╞══════════════╪═════════════╪════════╪════════════════════╪═════════════════╡
    │ 1            ┆ 5           ┆ 100    ┆ false              ┆ false           │
    │ 3            ┆ 8           ┆ 500    ┆ false              ┆ false           │
    │ 5            ┆ 10          ┆ 1200   ┆ false              ┆ true            │
    │ 0            ┆ 2           ┆ 50     ┆ false              ┆ false           │
    │ 10           ┆ 15          ┆ 2000   ┆ false              ┆ true            │
    └──────────────┴─────────────┴────────┴────────────────────┴─────────────────┘

    **Example 2: OR logic within a rule (high amount OR high velocity)**

    >>> or_transformer = RuleFeatures(
    ...     rules=[
    ...         [
    ...             {'column': 'amount', 'op': '>', 'value': 1000},
    ...             {'column': 'velocity_24h', 'op': '>=', 'value': 5}
    ...         ]
    ...     ],
    ...     rule_logic='or',
    ...     new_column_names=['is_high_risk'],
    ...     drop_conditions=True
    ... )
    >>> result = or_transformer.fit_transform(X)
    >>> result.select(['amount', 'velocity_24h', 'is_high_risk'])
    shape: (5, 3)
    ┌────────┬──────────────┬──────────────┐
    │ amount ┆ velocity_24h ┆ is_high_risk │
    │ ---    ┆ ---          ┆ ---          │
    │ i64    ┆ i64          ┆ bool         │
    ╞════════╪══════════════╪══════════════╡
    │ 100    ┆ 1            ┆ false        │
    │ 500    ┆ 3            ┆ false        │
    │ 1200   ┆ 5            ┆ true         │
    │ 50     ┆ 0            ┆ false        │
    │ 2000   ┆ 10           ┆ true         │
    └────────┴──────────────┴──────────────┘

    **Example 3: Multiple rules with different logic patterns**

    >>> complex_transformer = RuleFeatures(
    ...     rules=[
    ...         # New user AND high amount AND high velocity
    ...         [
    ...             {'column': 'is_new_user', 'op': '==', 'value': True},
    ...             {'column': 'amount', 'op': '>', 'value': 1000},
    ...             {'column': 'velocity_24h', 'op': '>', 'value': 3}
    ...         ],
    ...         # Very high velocity (simple rule)
    ...         [
    ...             {'column': 'velocity_24h', 'op': '>=', 'value': 10}
    ...         ]
    ...     ],
    ...     rule_logic='and',
    ...     new_column_names=['is_suspicious_new_user', 'is_extreme_velocity']
    ... )
    >>> result = complex_transformer.fit_transform(X)
    >>> result.select(['is_new_user', 'amount', 'velocity_24h',
    ...                'is_suspicious_new_user', 'is_extreme_velocity'])
    shape: (5, 5)
    ┌─────────────┬────────┬──────────────┬─────────────────────────┬──────────────────────┐
    │ is_new_user ┆ amount ┆ velocity_24h ┆ is_suspicious_new_user  ┆ is_extreme_velocity  │
    │ ---         ┆ ---    ┆ ---          ┆ ---                     ┆ ---                  │
    │ bool        ┆ i64    ┆ i64          ┆ bool                    ┆ bool                 │
    ╞═════════════╪════════╪══════════════╪═════════════════════════╪══════════════════════╡
    │ true        ┆ 100    ┆ 1            ┆ false                   ┆ false                │
    │ false       ┆ 500    ┆ 3            ┆ false                   ┆ false                │
    │ false       ┆ 1200   ┆ 5            ┆ false                   ┆ false                │
    │ true        ┆ 50     ┆ 0            ┆ false                   ┆ false                │
    │ false       ┆ 2000   ┆ 10           ┆ false                   ┆ true                 │
    └─────────────┴────────┴──────────────┴─────────────────────────┴──────────────────────┘

    Notes
    -----
    - Each rule group produces one boolean output column
    - All conditions within a rule are evaluated independently before combining
    - Missing values (null) in comparisons typically result in null/false
    - Creates intermediate boolean columns, so use drop_conditions=True for cleaner output
    - To create a single column from multiple rules with complex logic (AND of ORs),
      use this transformer to create intermediate columns, then combine them manually
    """

    rules: List[List[Dict[str, Any]]]
    rule_logic: Literal["and", "or"] = "and"
    new_column_names: List[str]
    drop_conditions: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("rules")
    def validate_rules(cls, rules):
        """Validate rules structure and condition dictionaries."""
        if not rules:
            raise ValueError("At least one rule must be specified")

        supported_ops = {">", "<", ">=", "<=", "==", "!="}

        for rule_idx, rule in enumerate(rules):
            if not rule:
                raise ValueError(
                    f"Rule {rule_idx}: must contain at least one condition"
                )

            for cond_idx, cond in enumerate(rule):
                # Check required keys
                if "column" not in cond:
                    raise ValueError(
                        f"Rule {rule_idx}, Condition {cond_idx}: 'column' key is required"
                    )
                if "op" not in cond:
                    raise ValueError(
                        f"Rule {rule_idx}, Condition {cond_idx}: 'op' key is required"
                    )

                # Validate operator
                if cond["op"] not in supported_ops:
                    raise ValueError(
                        f"Rule {rule_idx}, Condition {cond_idx}: operator '{cond['op']}' not supported. "
                        f"Supported operators: {supported_ops}"
                    )

                # Check that either value or other_column is specified (but not both)
                has_value = "value" in cond
                has_other_column = "other_column" in cond

                if not has_value and not has_other_column:
                    raise ValueError(
                        f"Rule {rule_idx}, Condition {cond_idx}: must specify either 'value' or 'other_column'"
                    )
                if has_value and has_other_column:
                    raise ValueError(
                        f"Rule {rule_idx}, Condition {cond_idx}: cannot specify both 'value' and 'other_column'"
                    )

        return rules

    @field_validator("new_column_names")
    def validate_new_column_names(cls, new_column_names, info):
        """Validate that new_column_names length matches rules length."""
        if not new_column_names:
            raise ValueError("At least one column name must be specified")

        # Check for duplicates
        if len(new_column_names) != len(set(new_column_names)):
            raise ValueError("Column names must be unique")

        # If rules are available in context, check length matches
        if "rules" in info.data:
            rules = info.data["rules"]
            if len(new_column_names) != len(rules):
                raise ValueError(
                    f"Number of column names ({len(new_column_names)}) must match "
                    f"number of rules ({len(rules)})"
                )

        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "RuleFeatures":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[Any], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        RuleFeatures
            Fitted transformer instance.
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating boolean features for each rule.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with new boolean features (one per rule).
        """
        all_condition_cols = []

        # Process each rule group
        for rule_idx, (rule, output_col_name) in enumerate(
            zip(self.rules, self.new_column_names)
        ):
            condition_cols = []

            # Create intermediate boolean column for each condition in this rule
            for cond_idx, cond in enumerate(rule):
                col_name = f"_rule_{rule_idx}_cond_{cond_idx}_{output_col_name}"
                column = cond["column"]
                op = cond["op"]

                # Build condition expression
                if "other_column" in cond:
                    # Column to column comparison
                    other_col = cond["other_column"]
                    expr = self._build_column_comparison(column, op, other_col)
                else:
                    # Column to scalar comparison
                    value = cond["value"]
                    expr = self._build_scalar_comparison(column, op, value)

                X = X.with_columns(expr.alias(col_name))
                condition_cols.append(col_name)
                all_condition_cols.append(col_name)

            # Combine conditions within this rule using rule_logic
            if len(condition_cols) == 1:
                # Single condition, no combining needed
                combined_expr = pl.col(condition_cols[0])
            elif self.rule_logic == "and":
                # AND: all conditions must be true
                combined_expr = pl.col(condition_cols[0])
                for col in condition_cols[1:]:
                    combined_expr = combined_expr & pl.col(col)
            else:  # 'or'
                # OR: at least one condition must be true
                combined_expr = pl.col(condition_cols[0])
                for col in condition_cols[1:]:
                    combined_expr = combined_expr | pl.col(col)

            # Create the output column for this rule
            X = X.with_columns(combined_expr.alias(output_col_name))

        # Optionally drop all intermediate condition columns
        if self.drop_conditions:
            X = X.drop(all_condition_cols)

        return X

    @staticmethod
    def _build_column_comparison(column: str, op: str, other_column: str) -> pl.Expr:
        """Build a Polars expression for column-to-column comparison."""
        if op == ">":
            return pl.col(column) > pl.col(other_column)
        elif op == "<":
            return pl.col(column) < pl.col(other_column)
        elif op == ">=":
            return pl.col(column) >= pl.col(other_column)
        elif op == "<=":
            return pl.col(column) <= pl.col(other_column)
        elif op == "==":
            return pl.col(column) == pl.col(other_column)
        elif op == "!=":
            return pl.col(column) != pl.col(other_column)
        else:
            raise ValueError(f"Unsupported operator: {op}")

    @staticmethod
    def _build_scalar_comparison(column: str, op: str, value: Any) -> pl.Expr:
        """Build a Polars expression for column-to-scalar comparison."""
        if op == ">":
            return pl.col(column) > value
        elif op == "<":
            return pl.col(column) < value
        elif op == ">=":
            return pl.col(column) >= value
        elif op == "<=":
            return pl.col(column) <= value
        elif op == "==":
            return pl.col(column) == value
        elif op == "!=":
            return pl.col(column) != value
        else:
            raise ValueError(f"Unsupported operator: {op}")
