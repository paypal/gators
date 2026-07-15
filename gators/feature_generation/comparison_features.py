from typing import Literal

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer

COMPARISON_OPERATORS = {
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "is_null": lambda x, y: x.is_null(),
    "is_not_null": lambda x, y: x.is_not_null(),
}

# Unary operators that only use the first column
UNARY_OPERATORS = {"is_null", "is_not_null"}


class ComparisonFeatures(_BaseTransformer):
    """
    Generates binary comparison features between pairs of columns, or unary null checks.

    Parameters
    ----------
    subset_a : list[str]
        List of column names for the left side of comparisons (or the only column for unary operators).
    subset_b : list[str]
        List of column names for the right side of comparisons.
        For unary operators ('is_null', 'is_not_null'), these values are ignored.
    operators : list[Literal[">", "<", ">=", "<=", "==", "!=", "is_null", "is_not_null"]]
        List of comparison operators to apply. Must match length of columns.
        Unary operators: 'is_null', 'is_not_null' (only use subset_a)
        Binary operators: '>', '<', '>=', '<=', '==', '!=' (use both subset_a and subset_b)
    drop_columns : bool, default=False
        Whether to drop the original columns after creating comparisons.

    Examples
    --------
    >>> from gators.feature_generation import ComparisonFeatures
    >>> import polars as pl

    >>> X ={'A': [10, 20, 30, 40],
    ...         'B': [15, 10, 30, 35],
    ...         'C': [5, 25, 20, 50]}
    >>> X = pl.DataFrame(X)

    **Example 1: Single comparison**

    >>> transformer = ComparisonFeatures(
    ...     subset_a=['A'],
    ...     subset_b=['B'],
    ...     operators=['>']
    ... )
    >>> transformer.fit(X)
    ComparisonFeatures(subset_a=['A'], subset_b=['B'], operators=['>'])
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 4)
    ┌──────┬──────┬──────┬─────────┐
    │  A   │  B   │  C   │ A_gt_B  │
    │ i64  │ i64  │ i64  │  bool   │
    ├──────┼──────┼──────┼─────────┤
    │  10  │  15  │  5   │  false  │
    │  20  │  10  │  25  │  true   │
    │  30  │  30  │  20  │  false  │
    │  40  │  35  │  50  │  true   │
    └──────┴──────┴──────┴─────────┘

    **Example 2: Multiple comparisons with different operators**

    >>> transformer = ComparisonFeatures(
    ...     subset_a=['A', 'B', 'A'],
    ...     subset_b=['B', 'C', 'C'],
    ...     operators=['>', '<', '>=']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 6)
    ┌──────┬──────┬──────┬─────────┬─────────┬─────────┐
    │  A   │  B   │  C   │ A_gt_B  │ B_lt_C  │ A_gte_C │
    │ i64  │ i64  │ i64  │  bool   │  bool   │  bool   │
    ├──────┼──────┼──────┼─────────┼─────────┼─────────┤
    │  10  │  15  │  5   │  false  │  false  │  true   │
    │  20  │  10  │  25  │  true   │  true   │  false  │
    │  30  │  30  │  20  │  false  │  false  │  true   │
    │  40  │  35  │  50  │  true   │  true   │  false  │
    └──────┴──────┴──────┴─────────┴─────────┴─────────┘

    **Example 3: Null checks (unary operators)**

    >>> data_with_nulls = pl.DataFrame({
    ...     'A': [10, None, 30, None],
    ...     'B': [15, 10, None, 35]
    ... })
    >>> transformer = ComparisonFeatures(
    ...     subset_a=['A', 'B'],
    ...     subset_b=['', ''],  # Ignored for unary operators
    ...     operators=['is_null', 'is_not_null']
    ... )
    >>> result = transformer.fit_transform(data_with_nulls)
    >>> result
    shape: (4, 4)
    ┌──────┬──────┬────────────┬────────────────┐
    │  A   │  B   │ A__is_null │ B__is_not_null │
    │ i64  │ i64  │  bool      │  bool          │
    ├──────┼──────┼────────────┼────────────────┤
    │  10  │  15  │  false     │  true          │
    │ null │  10  │  true      │  true          │
    │  30  │ null │  false     │  false         │
    │ null │  35  │  true      │  true          │
    └──────┴──────┴────────────┴────────────────┘

    **Example 4: With drop_columns=True**

    >>> transformer = ComparisonFeatures(
    ...     subset_a=['A'],
    ...     subset_b=['B'],
    ...     operators=['>'],
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 2)
    ┌──────┬─────────┐
    │  C   │ A_gt_B  │
    │ i64  │  bool   │
    ├──────┼─────────┤
    │  5   │  false  │
    │  25  │  true   │
    │  20  │  false  │
    │  50  │  true   │
    └──────┴─────────┘
    """

    subset_a: list[str]
    subset_b: list[str]
    operators: list[Literal[">", "<", ">=", "<=", "==", "!=", "is_null", "is_not_null"]]
    drop_columns: bool = False

    @field_validator("operators")
    def check_operators(cls, operators):
        for op in operators:
            if op not in COMPARISON_OPERATORS:
                raise ValueError(
                    f"Operator '{op}' is not supported. "
                    f"Supported operators: {list(COMPARISON_OPERATORS.keys())}"
                )
        return operators

    @field_validator("operators", mode="after")
    def check_lengths_match(cls, operators, info):
        subset_a = info.data.get("subset_a", [])
        subset_b = info.data.get("subset_b", [])

        if len(subset_a) != len(subset_b):
            raise ValueError(
                f"Length of subset_a ({len(subset_a)}) "
                f"must match length of subset_b ({len(subset_b)})"
            )

        if len(subset_a) != len(operators):
            raise ValueError(
                f"Length of subset_a ({len(subset_a)}) "
                f"must match length of operators ({len(operators)})"
            )

        return operators

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "ComparisonFeatures":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        ComparisonFeatures
            Fitted transformer instance.
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating comparison features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with comparison features.
        """
        new_columns = []
        to_drop = set()

        # Operator name mapping for column names
        op_names = {
            ">": "gt",
            "<": "lt",
            ">=": "gte",
            "<=": "lte",
            "==": "eq",
            "!=": "ne",
            "is_null": "is_null",
            "is_not_null": "is_not_null",
        }

        for col_a, col_b, op in zip(self.subset_a, self.subset_b, self.operators):
            op_name = op_names[op]

            # For unary operators, only use column_a in the name
            if op in UNARY_OPERATORS:
                new_col_name = f"{col_a}__{op_name}"
            else:
                new_col_name = f"{col_a}_{op_name}_{col_b}"

            comparison_expr = COMPARISON_OPERATORS[op](pl.col(col_a), pl.col(col_b)).alias(
                new_col_name
            )
            new_columns.append(comparison_expr)

            if self.drop_columns:
                to_drop.add(col_a)
                # Only drop col_b for binary operators
                if op not in UNARY_OPERATORS:
                    to_drop.add(col_b)

        X = X.with_columns(new_columns)

        if self.drop_columns and to_drop:
            X = X.drop(list(to_drop))

        return X
