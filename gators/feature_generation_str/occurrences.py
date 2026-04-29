from typing import Dict, List, Optional

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class Occurrences(_BaseTransformer):
    """
    Counts occurrences of specific substrings or characters in string columns.

    Creates count features showing how many times a substring appears,
    useful for tree-based models to split on frequency patterns.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of string columns to extract features from. If None, all string columns
        will be used.
    substrings : Dict[str, List[str]]
        Dictionary mapping column names to lists of substrings to count.
        For example: {"description": ["error", "warning", "success"]}
        will create 3 count features for the "description" column.
    case_sensitive : bool, default=False
        Whether substring matching should be case sensitive.
    drop_columns : bool, default=False
        Whether to drop the original string columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_str import Occurrences
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'log': ['Error: invalid input', 'Success: completed', 'Error: timeout Error', None],
    ...     'tags': ['#python #ml #data', '#python #java', '#ml #python', '']
    ... })

    **Example 1: Count specific keywords**

    >>> transformer = Occurrences(
    ...     subset=['log'],
    ...     substrings={'log': ['error', 'success', 'timeout']}
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (4, 5)
    ┌───────────────────────────┬────────────────────┬─────────────┬────────────────┬────────────────┐
    │ log                       ┆ tags               ┆ log__error  ┆ log__success   ┆ log__timeout   │
    │ ---                       ┆ ---                ┆ ---         ┆ ---            ┆ ---            │
    │ str                       ┆ str                ┆ i64         ┆ i64            ┆ i64            │
    ├───────────────────────────┼────────────────────┼─────────────┼────────────────┼────────────────┤
    │ Error: invalid input      ┆ #python #ml #data  ┆ 1           ┆ 0              ┆ 0              │
    │ Success: completed        ┆ #python #java      ┆ 0           ┆ 1              ┆ 0              │
    │ Error: timeout Error      ┆ #ml #python        ┆ 2           ┆ 0              ┆ 1              │
    │ null                      ┆                    ┆ 0           ┆ 0              ┆ 0              │
    └───────────────────────────┴────────────────────┴─────────────┴────────────────┴────────────────┘

    **Example 2: Count hashtags (case sensitive)**

    >>> transformer = Occurrences(
    ...     subset=['tags'],
    ...     substrings={'tags': ['#python', '#ml', '#java', '#data']},
    ...     case_sensitive=True
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: Multiple columns with drop**

    >>> transformer = Occurrences(
    ...     subset=['log', 'tags'],
    ...     substrings={
    ...         'log': ['error', 'success'],
    ...         'tags': ['#python', '#ml']
    ...     },
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    substrings: Dict[str, List[str]]
    case_sensitive: bool = False
    drop_columns: bool = False

    @field_validator("substrings")
    def check_substrings(cls, substrings):
        if not substrings:
            raise ValueError("substrings dictionary cannot be empty")
        for col, substrs in substrings.items():
            if not isinstance(substrs, list) or len(substrs) == 0:
                raise ValueError(f"Column '{col}' must have a non-empty list of substrings")
        return substrings

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Occurrences":
        """Fit the transformer by identifying string columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Occurrences
            Fitted transformer instance.
        """
        if not self.subset:
            # Use all columns that are in substrings and are string type
            string_cols = [
                col for col, dtype in X.schema.items() if dtype == pl.String or dtype == pl.Utf8
            ]
            self.subset = [col for col in self.substrings.keys() if col in string_cols]
        else:
            # Validate that specified columns are in substrings
            missing = set(self.subset) - set(self.substrings.keys())
            if missing:
                raise ValueError(
                    f"Columns {missing} are specified but not found in substrings dictionary"
                )
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by counting substring occurrences.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with substring count features.
        """
        if self.subset is None:
            return X

        new_columns = []

        for col in self.subset:
            if col not in self.substrings:
                continue

            col_expr = pl.col(col).fill_null("")

            # Group substrings by their safe names to handle duplicates
            safe_name_to_substrings: dict[str, list[str]] = {}
            for substring in self.substrings[col]:
                # Create safe feature name by replacing special chars
                safe_substring = (
                    substring.replace("#", "hash")
                    .replace("@", "at")
                    .replace(".", "dot")
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
                )

                if safe_substring not in safe_name_to_substrings:
                    safe_name_to_substrings[safe_substring] = []
                safe_name_to_substrings[safe_substring].append(substring)

            # Create count expressions for each unique safe name
            for safe_substring, substrings in safe_name_to_substrings.items():
                if self.case_sensitive:
                    # Sum counts for all substrings mapping to this safe name
                    count_exprs = []
                    for substring in substrings:
                        count_exprs.append(col_expr.str.count_matches(substring, literal=True))
                    # Sum all the counts
                    if len(count_exprs) == 1:
                        count_expr = count_exprs[0].alias(f"{col}__{safe_substring}")
                    else:
                        # Use fold to sum multiple expressions
                        total = count_exprs[0]
                        for expr in count_exprs[1:]:
                            total = total + expr
                        count_expr = total.alias(f"{col}__{safe_substring}")
                else:
                    # Count case-insensitive matches
                    import re

                    count_exprs = []
                    for substring in substrings:
                        escaped_substring = re.escape(substring)
                        pattern = f"(?i){escaped_substring}"
                        count_exprs.append(col_expr.str.count_matches(pattern))
                    # Sum all the counts
                    if len(count_exprs) == 1:
                        count_expr = count_exprs[0].alias(f"{col}__{safe_substring}")
                    else:
                        # Use fold to sum multiple expressions
                        total = count_exprs[0]
                        for expr in count_exprs[1:]:
                            total = total + expr
                        count_expr = total.alias(f"{col}__{safe_substring}")

                new_columns.append(count_expr)

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
