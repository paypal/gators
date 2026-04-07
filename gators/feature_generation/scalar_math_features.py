from typing import Any, Dict, List, Optional

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.base import BaseEstimator, TransformerMixin

# Operator symbol to name mapping for auto-generated column names
OPERATOR_NAMES = {
    "+": "plus",
    "-": "minus",
    "*": "mul",
    "/": "div",
    "**": "pow",
    "//": "floordiv",
    "%": "mod",
}


class ScalarMathFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates new features by applying mathematical operations between columns and scalar values.

    This transformer performs element-wise operations between a column and a scalar constant.
    Each operation creates one new feature column. For operations between multiple columns,
    use MathFeatures instead.

    **Use Cases:**

    - Unit conversions (days to years, meters to feet, Celsius to Fahrenheit)
    - Normalization (divide by constant, multiply by scaling factor)
    - Feature scaling (percentage calculation, ratio computation)
    - Offset adjustments (add/subtract baseline values)

    **When to Use:**

    - Need to apply arithmetic operations with fixed scalar values
    - Creating interpretable transformations (e.g., Age/365 for age_in_years)
    - Scaling features by known constants
    - Building feature sets for downstream models

    **When NOT to Use:**

    - Operations between multiple columns (use MathFeatures)
    - Need learned scaling (use StandardScaler, MinMaxScaler)
    - Complex mathematical functions (use DataFrame.with_columns directly)

    Parameters
    ----------
    operations : List[Dict[str, Any]]
        List of operation dictionaries. Each operation creates one new feature column.

        Each operation dictionary must contain:


        - 'column': str - Name of the column to operate on
        - 'op': str - Mathematical operator. Supported: '+', '-', '*', '/', '**', '//', '%'
        - 'scalar': float or int - Scalar value to use in the operation

        Examples::

            [
                {'column': 'Age', 'op': '/', 'scalar': 365},
                {'column': 'Price', 'op': '*', 'scalar': 1.1},
                {'column': 'Temperature', 'op': '+', 'scalar': 273.15}
            ]

    new_column_names : Optional[List[str]], default=None
        Names for the resulting feature columns. If provided, must have the same
        length as ``operations``. If None, column names are auto-generated in the format:
        ``{column}_{op_name}_{scalar}`` (e.g., 'Age_div_365', 'Price_mul_1.1')

        Operator name mapping: '+' -> 'plus', '-' -> 'minus', '*' -> 'mul',
        '/' -> 'div', '**' -> 'pow', '//' -> 'floordiv', '%' -> 'mod'

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation import ScalarMathFeatures

    >>> X ={
    ...     'Age': [25, 30, 45, 12, 65],
    ...     'Price': [100.0, 150.0, 200.0, 75.0, 300.0],
    ...     'Temperature': [20.0, 25.0, 15.0, 30.0, 22.0]
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Unit conversions with custom names**

    >>> transformer = ScalarMathFeatures(
    ...     operations=[
    ...         {'column': 'Age', 'op': '/', 'scalar': 365},
    ...         {'column': 'Temperature', 'op': '+', 'scalar': 273.15}
    ...     ],
    ...     new_column_names=['Age_years', 'Temperature_kelvin']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.select(['Age', 'Age_years', 'Temperature', 'Temperature_kelvin'])
    shape: (5, 4)
    ┌─────┬───────────┬─────────────┬───────────────────┐
    │ Age ┆ Age_years ┆ Temperature ┆ Temperature_kelvin│
    │ --- ┆ ---       ┆ ---         ┆ ---               │
    │ i64 ┆ f64       ┆ f64         ┆ f64               │
    ╞═════╪═══════════╪═════════════╪═══════════════════╡
    │ 25  ┆ 0.068493  ┆ 20.0        ┆ 293.15            │
    │ 30  ┆ 0.082192  ┆ 25.0        ┆ 298.15            │
    │ 45  ┆ 0.123288  ┆ 15.0        ┆ 288.15            │
    │ 12  ┆ 0.032877  ┆ 30.0        ┆ 303.15            │
    │ 65  ┆ 0.178082  ┆ 22.0        ┆ 295.15            │
    └─────┴───────────┴─────────────┴───────────────────┘

    **Example 2: Auto-generated column names**

    >>> auto_transformer = ScalarMathFeatures(
    ...     operations=[
    ...         {'column': 'Price', 'op': '*', 'scalar': 1.1},
    ...         {'column': 'Price', 'op': '/', 'scalar': 100}
    ...     ]
    ...     # new_column_names not specified - will be auto-generated
    ... )
    >>> result = auto_transformer.fit_transform(X)
    >>> result.select(['Price', 'Price_mul_1.1', 'Price_div_100'])
    shape: (5, 3)
    ┌───────┬──────────────┬───────────────┐
    │ Price ┆ Price_mul_1.1┆ Price_div_100 │
    │ ---   ┆ ---          ┆ ---           │
    │ f64   ┆ f64          ┆ f64           │
    ╞═══════╪══════════════╪═══════════════╡
    │ 100.0 ┆ 110.0        ┆ 1.0           │
    │ 150.0 ┆ 165.0        ┆ 1.5           │
    │ 200.0 ┆ 220.0        ┆ 2.0           │
    │ 75.0  ┆ 82.5         ┆ 0.75          │
    │ 300.0 ┆ 330.0        ┆ 3.0           │
    └───────┴──────────────┴───────────────┘

    **Example 3: Multiple operations (scaling, percentage, tax)**

    >>> multi_ops = ScalarMathFeatures(
    ...     operations=[
    ...         {'column': 'Price', 'op': '*', 'scalar': 1.2},  # 20% markup
    ...         {'column': 'Price', 'op': '/', 'scalar': 100},  # as percentage of 100
    ...         {'column': 'Age', 'op': '%', 'scalar': 10}      # age modulo 10
    ...     ],
    ...     new_column_names=['Price_with_tax', 'Price_pct', 'Age_decade_offset']
    ... )
    >>> result = multi_ops.fit_transform(X)
    >>> result.select(['Price', 'Price_with_tax', 'Price_pct', 'Age', 'Age_decade_offset'])
    shape: (5, 5)
    ┌───────┬────────────────┬───────────┬─────┬───────────────────┐
    │ Price ┆ Price_with_tax ┆ Price_pct ┆ Age ┆ Age_decade_offset │
    │ ---   ┆ ---            ┆ ---       ┆ --- ┆ ---               │
    │ f64   ┆ f64            ┆ f64       ┆ i64 ┆ i64               │
    ╞═══════╪════════════════╪═══════════╪═════╪═══════════════════╡
    │ 100.0 ┆ 120.0          ┆ 1.0       ┆ 25  ┆ 5                 │
    │ 150.0 ┆ 180.0          ┆ 1.5       ┆ 30  ┆ 0                 │
    │ 200.0 ┆ 240.0          ┆ 2.0       ┆ 45  ┆ 5                 │
    │ 75.0  ┆ 90.0           ┆ 0.75      ┆ 12  ┆ 2                 │
    │ 300.0 ┆ 360.0          ┆ 3.0       ┆ 65  ┆ 5                 │
    └───────┴────────────────┴───────────┴─────┴───────────────────┘

    **Example 4: Power and floor division**

    >>> power_ops = ScalarMathFeatures(
    ...     operations=[
    ...         {'column': 'Age', 'op': '**', 'scalar': 2},
    ...         {'column': 'Age', 'op': '//', 'scalar': 10}
    ...     ],
    ...     new_column_names=['Age_squared', 'Age_decade']
    ... )
    >>> result = power_ops.fit_transform(X)
    >>> result.select(['Age', 'Age_squared', 'Age_decade'])
    shape: (5, 3)
    ┌─────┬─────────────┬────────────┐
    │ Age ┆ Age_squared ┆ Age_decade │
    │ --- ┆ ---         ┆ ---        │
    │ i64 ┆ i64         ┆ i64        │
    ╞═════╪═════════════╪════════════╡
    │ 25  ┆ 625         ┆ 2          │
    │ 30  ┆ 900         ┆ 3          │
    │ 45  ┆ 2025        ┆ 4          │
    │ 12  ┆ 144         ┆ 1          │
    │ 65  ┆ 4225        ┆ 6          │
    └─────┴─────────────┴────────────┘

    Notes
    -----
    - Each operation produces exactly one new feature column
    - Auto-naming: If new_column_names is None, names are auto-generated as:
      `{column}_{op_name}_{scalar}` (e.g., 'Age_div_365')
    - Operations are applied element-wise to each row
    - Division by zero will result in inf or null values (Polars default behavior)
    - Can chain multiple ScalarMathFeatures transformers in a pipeline
    - For learned transformations, consider sklearn scalers instead

    See Also
    --------
    MathFeatures : For operations between multiple columns
    ConditionFeatures : For creating boolean features from conditions
    """

    operations: List[Dict[str, Any]]
    new_column_names: Optional[List[str]] = None
    _generated_column_names: List[str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("operations")
    def validate_operations(cls, operations):
        """Validate operations structure."""
        if not operations:
            raise ValueError("At least one operation must be specified")

        supported_ops = {"+", "-", "*", "/", "**", "//", "%"}

        for idx, op_dict in enumerate(operations):
            # Check required keys
            if "column" not in op_dict:
                raise ValueError(f"Operation {idx}: 'column' key is required")
            if "op" not in op_dict:
                raise ValueError(f"Operation {idx}: 'op' key is required")
            if "scalar" not in op_dict:
                raise ValueError(f"Operation {idx}: 'scalar' key is required")

            # Validate operator
            if op_dict["op"] not in supported_ops:
                raise ValueError(
                    f"Operation {idx}: operator '{op_dict['op']}' not supported. "
                    f"Supported operators: {supported_ops}"
                )

            # Validate scalar is numeric
            scalar = op_dict["scalar"]
            if not isinstance(scalar, (int, float)):
                raise ValueError(
                    f"Operation {idx}: 'scalar' must be numeric (int or float), "
                    f"got {type(scalar).__name__}"
                )

        return operations

    @field_validator("new_column_names")
    def validate_new_column_names(cls, new_column_names, info):
        """Validate that new_column_names length matches operations length."""
        # If None, auto-naming will be used
        if new_column_names is None:
            return new_column_names

        if not new_column_names:
            raise ValueError("Column names list cannot be empty (use None for auto-naming)")

        # Check for duplicates
        if len(new_column_names) != len(set(new_column_names)):
            raise ValueError("Column names must be unique")

        # If operations are available in context, check length matches
        if "operations" in info.data:
            operations = info.data["operations"]
            if len(new_column_names) != len(operations):
                raise ValueError(
                    f"Number of column names ({len(new_column_names)}) must match "
                    f"number of operations ({len(operations)})"
                )

        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[Any] = None) -> "ScalarMathFeatures":
        """Fit the transformer by generating column names if not provided.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[Any], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        ScalarMathFeatures
            Fitted transformer instance.
        """
        # Generate column names if not provided
        if self.new_column_names is None:
            self._generated_column_names = []
            for op_dict in self.operations:
                column = op_dict["column"]
                op = op_dict["op"]
                scalar = op_dict["scalar"]
                op_name = OPERATOR_NAMES[op]

                # Format scalar to avoid unnecessary decimals
                if isinstance(scalar, float) and scalar == int(scalar):
                    formatted_scalar = int(scalar)
                else:
                    formatted_scalar = scalar

                col_name = f"{column}_{op_name}_{formatted_scalar}"
                self._generated_column_names.append(col_name)
        else:
            self._generated_column_names = self.new_column_names

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating new features from scalar operations.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with new computed features (one per operation).
        """
        new_columns = []

        # Process each operation
        for op_dict, output_col_name in zip(self.operations, self._generated_column_names):
            column = op_dict["column"]
            op = op_dict["op"]
            scalar = op_dict["scalar"]

            # Build operation expression
            expr = self._build_operation(column, op, scalar)
            new_columns.append(expr.alias(output_col_name))

        # Add all new columns at once
        X = X.with_columns(new_columns)

        return X

    @staticmethod
    def _build_operation(column: str, op: str, scalar: float) -> pl.Expr:
        """Build a Polars expression for column-scalar operation."""
        if op == "+":
            return pl.col(column) + scalar
        elif op == "-":
            return pl.col(column) - scalar
        elif op == "*":
            return pl.col(column) * scalar
        elif op == "/":
            return pl.col(column) / scalar
        elif op == "**":
            return pl.col(column) ** scalar
        elif op == "//":
            return pl.col(column) // scalar
        elif op == "%":
            return pl.col(column) % scalar
        else:
            raise ValueError(f"Unsupported operator: {op}")
