from typing import Any, Dict, List, Optional

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.base import BaseEstimator, TransformerMixin

OPERATOR_NAMES = {
    ">": "gt",
    "<": "lt",
    ">=": "gte",
    "<=": "lte",
    "==": "eq",
    "!=": "ne",
    "is_null": "is_null",
    "is_not_null": "is_not_null",
}

# Unary operators that only use the column (no value or other_column needed)
UNARY_OPERATORS = {"is_null", "is_not_null"}


class ConditionFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Creates multiple independent boolean features, one for each condition.

    This transformer is designed for creating simple boolean flags without combination logic.
    Each condition produces exactly one boolean output column. For combining multiple
    conditions with AND/OR logic, use RuleFeatures instead.

    **Use Cases:**

    - Create simple boolean flags (is_adult, is_weekend, is_premium, etc.)
    - Materialize threshold-based features (is_high_value, is_frequent_user)
    - Feature engineering: Generate independent indicator variables
    - Fraud detection: Create simple risk flags before combining them

    **When to Use:**

    - Need multiple independent boolean columns
    - Each condition stands alone (no AND/OR combination needed)
    - Want cleaner API than RuleFeatures for simple cases
    - Building feature sets for downstream transformers

    **When NOT to Use:**

    - Need to combine conditions with AND/OR (use RuleFeatures)
    - One-off exploratory analysis (use Polars native expressions)
    - Very simple cases with 1-2 conditions (just use .with_columns())

    Parameters
    ----------
    conditions : List[Dict[str, Any]]
        List of condition dictionaries. Each condition creates one boolean output column.

        Each condition dictionary must contain:

        - 'column': str - Name of the column to evaluate
        - 'op': str - Comparison operator. Supported:

          * Binary: '>', '<', '>=', '<=', '==', '!=' (require 'value' or 'other_column')
          * Unary: 'is_null', 'is_not_null' (no 'value' or 'other_column' needed)

        - 'value': Any (optional) - Scalar value to compare the column against
        - 'other_column': str (optional) - Name of another column to compare against

        For binary operators: Either 'value' or 'other_column' must be specified, but not both.
        For unary operators: Neither 'value' nor 'other_column' should be specified.

        Examples::

            # Simple conditions:
            [
                {'column': 'age', 'op': '>=', 'value': 18},
                {'column': 'amount', 'op': '>', 'value': 1000}
            ]

            # Column comparison:
            [
                {'column': 'velocity_24h', 'op': '>', 'other_column': 'velocity_7d'}
            ]

            # Null checks:
            [
                {'column': 'age', 'op': 'is_null'},
                {'column': 'email', 'op': 'is_not_null'}
            ]

    new_column_names : Optional[List[str]], default=None
        Names for the resulting boolean feature columns. If provided, must have the same
        length as ``conditions``. If None, column names are auto-generated in the format:

        - Scalar comparison: ``{column}_{op_name}_{value}`` (e.g., 'age_gte_18')
        - Column comparison: ``{column}_{op_name}_{other_column}`` (e.g., 'velocity_24h_gt_velocity_7d')
        - Unary operation: ``{column}__{op_name}`` (e.g., 'age__is_null')

        Operator name mapping:

        - '>' -> 'gt'
        - '<' -> 'lt'
        - '>=' -> 'gte'
        - '<=' -> 'lte'
        - '==' -> 'eq'
        - '!=' -> 'ne'
        - 'is_null' -> 'is_null'
        - 'is_not_null' -> 'is_not_null'

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation import ConditionFeatures

    >>> X ={
    ...     'age': [15, 25, 30, 17, 45],
    ...     'amount': [100, 1500, 500, 200, 2000],
    ...     'family_size': [1, 3, 1, 4, 2],
    ...     'fare': [50, 75, 30, 100, 80]
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Create simple boolean flags**

    >>> transformer = ConditionFeatures(
    ...     conditions=[
    ...         {'column': 'age', 'op': '>=', 'value': 18},
    ...         {'column': 'amount', 'op': '>', 'value': 1000},
    ...         {'column': 'family_size', 'op': '==', 'value': 1}
    ...     ],
    ...     new_column_names=['is_adult', 'is_high_amount', 'is_alone']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.select(['age', 'amount', 'family_size', 'is_adult', 'is_high_amount', 'is_alone'])
    shape: (5, 6)
    ┌─────┬────────┬─────────────┬──────────┬─────────────────┬──────────┐
    │ age ┆ amount ┆ family_size ┆ is_adult ┆ is_high_amount  ┆ is_alone │
    │ --- ┆ ---    ┆ ---         ┆ ---      ┆ ---             ┆ ---      │
    │ i64 ┆ i64    ┆ i64         ┆ bool     ┆ bool            ┆ bool     │
    ╞═════╪════════╪═════════════╪══════════╪═════════════════╪══════════╡
    │ 15  ┆ 100    ┆ 1           ┆ false    ┆ false           ┆ true     │
    │ 25  ┆ 1500   ┆ 3           ┆ true     ┆ true            ┆ false    │
    │ 30  ┆ 500    ┆ 1           ┆ true     ┆ false           ┆ true     │
    │ 17  ┆ 200    ┆ 4           ┆ false    ┆ false           ┆ false    │
    │ 45  ┆ 2000   ┆ 2           ┆ true     ┆ true            ┆ false    │
    └─────┴────────┴─────────────┴──────────┴─────────────────┴──────────┘

    **Example 2: Column-to-column comparison**

    >>> fare_X ={
    ...     'fare': [50.0, 100.0, 30.0, 200.0, 80.0],
    ...     'fare_per_person': [50.0, 33.3, 30.0, 50.0, 40.0]
    ... }
    >>> fare_X = pl.DataFrame(fare_data)
    >>> fare_transformer = ConditionFeatures(
    ...     conditions=[
    ...         {'column': 'fare', 'op': '>', 'value': 100},
    ...         {'column': 'fare_per_person', 'op': '>', 'other_column': 'fare'}
    ...     ],
    ...     new_column_names=['is_expensive', 'paid_more_per_person']
    ... )
    >>> result = fare_transformer.fit_transform(fare_X)
    >>> result
    shape: (5, 4)
    ┌───────┬──────────────────┬──────────────┬──────────────────────┐
    │ fare  ┆ fare_per_person  ┆ is_expensive ┆ paid_more_per_person │
    │ ---   ┆ ---              ┆ ---          ┆ ---                  │
    │ f64   ┆ f64              ┆ bool         ┆ bool                 │
    ╞═══════╪══════════════════╪══════════════╪══════════════════════╡
    │ 50.0  ┆ 50.0             ┆ false        ┆ false                │
    │ 100.0 ┆ 33.3             ┆ false        ┆ false                │
    │ 30.0  ┆ 30.0             ┆ false        ┆ false                │
    │ 200.0 ┆ 50.0             ┆ true         ┆ false                │
    │ 80.0  ┆ 40.0             ┆ false        ┆ false                │
    └───────┴──────────────────┴──────────────┴──────────────────────┘

    **Example 3: Titanic-style feature engineering**

    >>> titanic_X ={
    ...     'Age': [22.0, 38.0, 26.0, 35.0, 12.0],
    ...     'Pclass': [3, 1, 3, 1, 3],
    ...     'SibSp': [1, 1, 0, 1, 0],
    ...     'Parch': [0, 0, 0, 0, 1]
    ... }
    >>> titanic_X = pl.DataFrame(titanic_data)
    >>> # First add family_size
    >>> titanic_X = titanic_X.with_columns(
    ...     (pl.col('SibSp') + pl.col('Parch')).alias('family_size')
    ... )
    >>> titanic_transformer = ConditionFeatures(
    ...     conditions=[
    ...         {'column': 'Age', 'op': '<', 'value': 18},
    ...         {'column': 'Pclass', 'op': '==', 'value': 1},
    ...         {'column': 'family_size', 'op': '==', 'value': 0}
    ...     ],
    ...     new_column_names=['is_child', 'is_first_class', 'is_alone']
    ... )
    >>> result = titanic_transformer.fit_transform(titanic_X)
    >>> result.select(['Age', 'Pclass', 'family_size', 'is_child', 'is_first_class', 'is_alone'])
    shape: (5, 6)
    ┌──────┬────────┬─────────────┬──────────┬────────────────┬──────────┐
    │ Age  ┆ Pclass ┆ family_size ┆ is_child ┆ is_first_class ┆ is_alone │
    │ ---  ┆ ---    ┆ ---         ┆ ---      ┆ ---            ┆ ---      │
    │ f64  ┆ i64    ┆ i64         ┆ bool     ┆ bool           ┆ bool     │
    ╞══════╪════════╪═════════════╪══════════╪════════════════╪══════════╡
    │ 22.0 ┆ 3      ┆ 1           ┆ false    ┆ false          ┆ false    │
    │ 38.0 ┆ 1      ┆ 1           ┆ false    ┆ true           ┆ false    │
    │ 26.0 ┆ 3      ┆ 0           ┆ false    ┆ false          ┆ true     │
    │ 35.0 ┆ 1      ┆ 1           ┆ false    ┆ true           ┆ false    │
    │ 12.0 ┆ 3      ┆ 1           ┆ true     ┆ false          ┆ false    │
    └──────┴────────┴─────────────┴──────────┴────────────────┴──────────┘

    **Example 4: Auto-generated column names**

    >>> auto_transformer = ConditionFeatures(
    ...     conditions=[
    ...         {'column': 'age', 'op': '>=', 'value': 18},
    ...         {'column': 'amount', 'op': '>', 'value': 1000},
    ...         {'column': 'family_size', 'op': '==', 'value': 1}
    ...     ]
    ...     # new_column_names not specified - will be auto-generated
    ... )
    >>> result = auto_transformer.fit_transform(X)
    >>> result.select(['age', 'amount', 'family_size', 'age_gte_18', 'amount_gt_1000', 'family_size_eq_1'])
    shape: (5, 6)
    ┌─────┬────────┬─────────────┬────────────┬────────────────┬──────────────────┐
    │ age ┆ amount ┆ family_size ┆ age_gte_18 ┆ amount_gt_1000 ┆ family_size_eq_1 │
    │ --- ┆ ---    ┆ ---         ┆ ---        ┆ ---            ┆ ---              │
    │ i64 ┆ i64    ┆ i64         ┆ bool       ┆ bool           ┆ bool             │
    ╞═════╪════════╪═════════════╪════════════╪════════════════╪══════════════════╡
    │ 15  ┆ 100    ┆ 1           ┆ false      ┆ false          ┆ true             │
    │ 25  ┆ 1500   ┆ 3           ┆ true       ┆ true           ┆ false            │
    │ 30  ┆ 500    ┆ 1           ┆ true       ┆ false          ┆ true             │
    │ 17  ┆ 200    ┆ 4           ┆ false      ┆ false          ┆ false            │
    │ 45  ┆ 2000   ┆ 2           ┆ true       ┆ true           ┆ false            │
    └─────┴────────┴─────────────┴────────────┴────────────────┴──────────────────┘

    **Example 5: Null checks (unary operators)**

    >>> data_with_nulls = {
    ...     'age': [25, None, 30, 17, None],
    ...     'email': ['a@test.com', 'b@test.com', None, 'd@test.com', None],
    ...     'amount': [100, 1500, 500, 200, 2000]
    ... }
    >>> X_nulls = pl.DataFrame(data_with_nulls)
    >>> null_transformer = ConditionFeatures(
    ...     conditions=[
    ...         {'column': 'age', 'op': 'is_null'},
    ...         {'column': 'email', 'op': 'is_not_null'},
    ...         {'column': 'amount', 'op': '>', 'value': 1000}
    ...     ],
    ...     new_column_names=['age_missing', 'has_email', 'is_high_amount']
    ... )
    >>> result = null_transformer.fit_transform(X_nulls)
    >>> result
    shape: (5, 6)
    ┌──────┬─────────────┬────────┬─────────────┬───────────┬─────────────────┐
    │ age  ┆ email       ┆ amount ┆ age_missing ┆ has_email ┆ is_high_amount  │
    │ ---  ┆ ---         ┆ ---    ┆ ---         ┆ ---       ┆ ---             │
    │ i64  ┆ str         ┆ i64    ┆ bool        ┆ bool      ┆ bool            │
    ╞══════╪═════════════╪════════╪═════════════╪═══════════╪═════════════════╡
    │ 25   ┆ a@test.com  ┆ 100    ┆ false       ┆ true      ┆ false           │
    │ null ┆ b@test.com  ┆ 1500   ┆ true        ┆ true      ┆ true            │
    │ 30   ┆ null        ┆ 500    ┆ false       ┆ false     ┆ false           │
    │ 17   ┆ d@test.com  ┆ 200    ┆ false       ┆ true      ┆ false           │
    │ null ┆ null        ┆ 2000   ┆ true        ┆ false     ┆ true            │
    └──────┴─────────────┴────────┴─────────────┴───────────┴─────────────────┘

    Notes
    -----
    - Each condition produces exactly one independent boolean column
    - Auto-naming: If new_column_names is None, names are auto-generated as:
      * Scalar: `{column}_{op_name}_{value}` (e.g., 'age_gte_18')
      * Column-to-column: `{column}_{op_name}_{other_column}` (e.g., 'velocity_24h_gt_velocity_7d')
      * Unary: `{column}__{op_name}` (e.g., 'age__is_null')
    - No combination logic - use RuleFeatures if you need AND/OR
    - Simpler API than RuleFeatures for common use cases
    - Missing values (null) in comparisons typically result in null/false
    - Unary operators 'is_null' and 'is_not_null' explicitly check for null values
    - Can be used as preprocessing step before RuleFeatures for complex logic

    See Also
    --------
    RuleFeatures : For combining multiple conditions with AND/OR logic
    """

    conditions: List[Dict[str, Any]]
    new_column_names: Optional[List[str]] = None
    _generated_column_names: List[str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("conditions")
    def validate_conditions(cls, conditions):
        """Validate conditions structure."""
        if not conditions:
            raise ValueError("At least one condition must be specified")

        supported_ops = {">", "<", ">=", "<=", "==", "!=", "is_null", "is_not_null"}

        for idx, cond in enumerate(conditions):
            # Check required keys
            if "column" not in cond:
                raise ValueError(f"Condition {idx}: 'column' key is required")
            if "op" not in cond:
                raise ValueError(f"Condition {idx}: 'op' key is required")

            # Validate operator
            if cond["op"] not in supported_ops:
                raise ValueError(
                    f"Condition {idx}: operator '{cond['op']}' not supported. "
                    f"Supported operators: {supported_ops}"
                )

            # Check value/other_column requirements based on operator type
            op = cond["op"]
            has_value = "value" in cond
            has_other_column = "other_column" in cond

            if op in UNARY_OPERATORS:
                # Unary operators should not have value or other_column
                if has_value or has_other_column:
                    raise ValueError(
                        f"Condition {idx}: unary operator '{op}' should not have "
                        f"'value' or 'other_column' specified"
                    )
            else:
                # Binary operators need exactly one of value or other_column
                if not has_value and not has_other_column:
                    raise ValueError(
                        f"Condition {idx}: must specify either 'value' or 'other_column'"
                    )
                if has_value and has_other_column:
                    raise ValueError(
                        f"Condition {idx}: cannot specify both 'value' and 'other_column'"
                    )

        return conditions

    @field_validator("new_column_names")
    def validate_new_column_names(cls, new_column_names, info):
        """Validate that new_column_names length matches conditions length."""
        # If None, auto-naming will be used
        if new_column_names is None:
            return new_column_names

        if not new_column_names:
            raise ValueError("Column names list cannot be empty (use None for auto-naming)")

        # Check for duplicates
        if len(new_column_names) != len(set(new_column_names)):
            raise ValueError("Column names must be unique")

        # If conditions are available in context, check length matches
        if "conditions" in info.data:
            conditions = info.data["conditions"]
            if len(new_column_names) != len(conditions):
                raise ValueError(
                    f"Number of column names ({len(new_column_names)}) must match "
                    f"number of conditions ({len(conditions)})"
                )

        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "ConditionFeatures":
        """Fit the transformer by generating column names if not provided.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[Any], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        ConditionFeatures
            Fitted transformer instance.
        """
        # Generate column names if not provided
        if self.new_column_names is None:
            self._generated_column_names = []
            for cond in self.conditions:
                column = cond["column"]
                op = cond["op"]
                op_name = OPERATOR_NAMES[op]

                if op in UNARY_OPERATORS:
                    # Unary operators: just column and operator
                    col_name = f"{column}__{op_name}"
                elif "other_column" in cond:
                    # Column-to-column comparison
                    other_col = cond["other_column"]
                    col_name = f"{column}_{op_name}_{other_col}"
                else:
                    # Scalar comparison
                    value = cond["value"]
                    # Format value to avoid unnecessary decimals
                    if isinstance(value, (int, float)) and value == int(value):
                        formatted_value = int(value)
                    else:
                        formatted_value = value
                    col_name = f"{column}_{op_name}_{formatted_value}"

                self._generated_column_names.append(col_name)
        else:
            self._generated_column_names = self.new_column_names

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating boolean features for each condition.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with new boolean features (one per condition).
        """
        new_columns = []

        # Process each condition
        for cond, output_col_name in zip(self.conditions, self._generated_column_names):
            column = cond["column"]
            op = cond["op"]

            # Build condition expression
            if op in UNARY_OPERATORS:
                # Unary operation (is_null, is_not_null)
                expr = self._build_unary_operation(column, op)
            elif "other_column" in cond:
                # Column to column comparison
                other_col = cond["other_column"]
                expr = self._build_column_comparison(column, op, other_col)
            else:
                # Column to scalar comparison
                value = cond["value"]
                expr = self._build_scalar_comparison(column, op, value)

            new_columns.append(expr.alias(output_col_name))

        # Add all new columns at once
        X = X.with_columns(new_columns)

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
    def _build_unary_operation(column: str, op: str) -> pl.Expr:
        """Build a Polars expression for unary operations (is_null, is_not_null)."""
        if op == "is_null":
            return pl.col(column).is_null()
        elif op == "is_not_null":
            return pl.col(column).is_not_null()
        else:
            raise ValueError(f"Unsupported unary operator: {op}")

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
