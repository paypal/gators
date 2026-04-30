from typing import List, Optional

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class PatternDetector(_BaseTransformer):
    """
    Detects common patterns in string columns (emails, URLs, phone numbers, etc.).

    Creates boolean features indicating whether strings match common patterns,
    useful for tree-based models to branch on data format and validity.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of string columns to extract features from. If None, all string columns
        will be used.
    patterns : List[str], default=["is_numeric", "is_email", "is_url", "is_phone"]
        Patterns to detect. Options:

        - "is_numeric": Contains only digits (with decimal/negative)
        - "is_email": Matches email pattern (basic check)
        - "is_url": Matches URL pattern (http/https)
        - "is_phone": Matches phone number pattern
        - "is_alphanumeric": Contains only letters and digits
        - "is_alpha": Contains only letters
        - "has_http": Contains http:// or https://
        - "has_www": Contains www.
        - "has_at": Contains @ symbol
    drop_columns : bool, default=False
        Whether to drop the original string columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_str import PatternDetector
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'contact': ['user@test.com', 'https://site.com', '555-1234', 'Hello World', None],
    ...     'code': ['ABC123', '999', 'test@email', 'XYZ', '']
    ... })

    **Example 1: Email and URL detection**

    >>> transformer = PatternDetector(
    ...     subset=['contact'],
    ...     patterns=['is_email', 'is_url', 'is_phone']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (5, 5)
    ┌────────────────────┬─────────┬───────────────────┬──────────────────┬───────────────────┐
    │ contact            ┆ code    ┆ contact__is_email ┆ contact__is_url  ┆ contact__is_phone │
    │ ---                ┆ ---     ┆ ---               ┆ ---              ┆ ---               │
    │ str                ┆ str     ┆ bool              ┆ bool             ┆ bool              │
    ├────────────────────┼─────────┼───────────────────┼──────────────────┼───────────────────┤
    │ user@test.com      ┆ ABC123  ┆ true              ┆ false            ┆ false             │
    │ https://site.com   ┆ 999     ┆ false             ┆ true             ┆ false             │
    │ 555-1234           ┆ test@e… ┆ false             ┆ false            ┆ true              │
    │ Hello World        ┆ XYZ     ┆ false             ┆ false            ┆ false             │
    │ null               ┆         ┆ false             ┆ false            ┆ false             │
    └────────────────────┴─────────┴───────────────────┴──────────────────┴───────────────────┘

    **Example 2: Numeric and alphanumeric detection**

    >>> transformer = PatternDetector(
    ...     subset=['code'],
    ...     patterns=['is_numeric', 'is_alphanumeric', 'is_alpha']
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: URL component detection**

    >>> transformer = PatternDetector(
    ...     subset=['contact'],
    ...     patterns=['has_http', 'has_www', 'has_at'],
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    patterns: List[str] = ["is_numeric", "is_email", "is_url", "is_phone"]
    drop_columns: bool = False

    @field_validator("patterns")
    def check_patterns(cls, patterns):
        valid_patterns = [
            "is_numeric",
            "is_email",
            "is_url",
            "is_phone",
            "is_alphanumeric",
            "is_alpha",
            "has_http",
            "has_www",
            "has_at",
        ]
        for pattern in patterns:
            if pattern not in valid_patterns:
                raise ValueError(
                    f"Pattern '{pattern}' is not supported. "
                    f"Supported patterns: {valid_patterns}"
                )
        return patterns

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "PatternDetector":
        """Fit the transformer by identifying string columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        PatternDetector
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype == pl.String or dtype == pl.Utf8
            ]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating pattern detection features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with pattern detection features.
        """
        if self.subset is None:
            return X

        new_columns = []

        for col in self.subset:
            col_expr = pl.col(col).fill_null("")

            if "is_numeric" in self.patterns:
                # Matches numbers (integer or float, with optional negative sign)
                is_numeric = (
                    col_expr.str.contains(r"^-?\d+\.?\d*$")
                    .fill_null(False)
                    .alias(f"{col}__is_numeric")
                )
                new_columns.append(is_numeric)

            if "is_email" in self.patterns:
                # Basic email pattern: something@something.something
                is_email = (
                    col_expr.str.contains(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
                    .fill_null(False)
                    .alias(f"{col}__is_email")
                )
                new_columns.append(is_email)

            if "is_url" in self.patterns:
                # URL pattern: http:// or https:// followed by domain
                is_url = (
                    col_expr.str.contains(r"^https?://[^\s]+$")
                    .fill_null(False)
                    .alias(f"{col}__is_url")
                )
                new_columns.append(is_url)

            if "is_phone" in self.patterns:
                # Phone pattern: various formats like 555-1234, (555) 123-4567, 5551234567
                is_phone = (
                    col_expr.str.contains(r"^[\d\s\-\(\)\.+]+$")
                    .fill_null(False)
                    .alias(f"{col}__is_phone")
                )
                new_columns.append(is_phone)

            if "is_alphanumeric" in self.patterns:
                # Only letters and digits
                is_alphanum = (
                    col_expr.str.contains(r"^[a-zA-Z0-9]+$")
                    .fill_null(False)
                    .alias(f"{col}__is_alphanumeric")
                )
                new_columns.append(is_alphanum)

            if "is_alpha" in self.patterns:
                # Only letters
                is_alpha = (
                    col_expr.str.contains(r"^[a-zA-Z]+$").fill_null(False).alias(f"{col}__is_alpha")
                )
                new_columns.append(is_alpha)

            if "has_http" in self.patterns:
                # Contains http:// or https://
                has_http = (
                    col_expr.str.contains(r"https?://").fill_null(False).alias(f"{col}__has_http")
                )
                new_columns.append(has_http)

            if "has_www" in self.patterns:
                # Contains www.
                has_www = col_expr.str.contains(r"www\.").fill_null(False).alias(f"{col}__has_www")
                new_columns.append(has_www)

            if "has_at" in self.patterns:
                # Contains @ symbol
                has_at = col_expr.str.contains(r"@").fill_null(False).alias(f"{col}__has_at")
                new_columns.append(has_at)

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
