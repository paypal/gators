from typing import List, Optional

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class CharacterStatistics(_BaseTransformer):
    """
    Generates character-level statistical features from string columns.

    Counts various character types (digits, letters, spaces, etc.) which are
    particularly useful for tree-based models to identify patterns in text data.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of string columns to extract features from. If None, all string columns
        will be used.
    features : List[str], default=["n_digits", "n_letters", "n_uppercase", "n_lowercase", "n_spaces", "n_special"]
        Character statistics to generate. Options:

        - "n_digits": Count of digit characters (0-9)
        - "n_letters": Count of alphabetic characters (a-z, A-Z)
        - "n_uppercase": Count of uppercase letters
        - "n_lowercase": Count of lowercase letters
        - "n_spaces": Count of space characters
        - "n_special": Count of special characters (punctuation, symbols)
        - "n_unique_chars": Count of unique characters
        - "ratio_uppercase": Ratio of uppercase to total letters
        - "ratio_digits": Ratio of digits to total length
        - "ratio_special": Ratio of special chars to total length
    drop_columns : bool, default=False
        Whether to drop the original string columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_str import CharacterStatistics
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'text': ['Hello123', 'WORLD!!!', 'Test 99', ''],
    ...     'email': ['user@test.com', 'ADMIN@SITE.ORG', 'test', None]
    ... })

    **Example 1: Basic character counts**

    >>> transformer = CharacterStatistics(
    ...     subset=['text'],
    ...     features=['n_digits', 'n_letters', 'n_uppercase']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result)
    shape: (4, 5)
    ┌──────────┬────────────────┬───────────────┬───────────────────┬─────────────────────┐
    │ text     ┆ email          ┆ text__n_digit ┆ text__n_letters   ┆ text__n_uppercase   │
    │ ---      ┆ ---            ┆ s             ┆ ---               ┆ ---                 │
    │ str      ┆ str            ┆ ---           ┆ i64               ┆ i64                 │
    │          ┆                ┆ i64           ┆                   ┆                     │
    ├──────────┼────────────────┼───────────────┼───────────────────┼─────────────────────┤
    │ Hello123 ┆ user@test.com  ┆ 3             ┆ 5                 ┆ 1                   │
    │ WORLD!!! ┆ ADMIN@SITE.ORG ┆ 0             ┆ 5                 ┆ 5                   │
    │ Test 99  ┆ test           ┆ 2             ┆ 4                 ┆ 1                   │
    │          ┆ null           ┆ 0             ┆ 0                 ┆ 0                   │
    └──────────┴────────────────┴───────────────┴───────────────────┴─────────────────────┘

    **Example 2: Ratio features**

    >>> transformer = CharacterStatistics(
    ...     subset=['text', 'email'],
    ...     features=['ratio_uppercase', 'ratio_digits', 'ratio_special']
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: All features with drop_columns**

    >>> transformer = CharacterStatistics(
    ...     subset=['text'],
    ...     features=['n_digits', 'n_letters', 'n_spaces', 'n_special', 'n_unique_chars'],
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    features: List[str] = [
        "n_digits",
        "n_letters",
        "n_uppercase",
        "n_lowercase",
        "n_spaces",
        "n_special",
    ]
    drop_columns: bool = False

    @field_validator("features")
    def check_features(cls, features):
        valid_features = [
            "n_digits",
            "n_letters",
            "n_uppercase",
            "n_lowercase",
            "n_spaces",
            "n_special",
            "n_unique_chars",
            "ratio_uppercase",
            "ratio_digits",
            "ratio_special",
        ]
        for feature in features:
            if feature not in valid_features:
                raise ValueError(
                    f"Feature '{feature}' is not supported. "
                    f"Supported features: {valid_features}"
                )
        return features

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CharacterStatistics":
        """Fit the transformer by identifying string columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        CharacterStatistics
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype == pl.String or dtype == pl.Utf8
            ]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating character statistics features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with character statistics features.
        """
        if self.subset is None:
            return X

        new_columns = []

        for col in self.subset:
            col_expr = pl.col(col).fill_null("")

            # Count features
            if "n_digits" in self.features:
                n_digits = col_expr.str.count_matches(r"\d").alias(f"{col}__n_digits")
                new_columns.append(n_digits)

            if "n_letters" in self.features:
                n_letters = col_expr.str.count_matches(r"[a-zA-Z]").alias(f"{col}__n_letters")
                new_columns.append(n_letters)

            if "n_uppercase" in self.features:
                n_uppercase = col_expr.str.count_matches(r"[A-Z]").alias(f"{col}__n_uppercase")
                new_columns.append(n_uppercase)

            if "n_lowercase" in self.features:
                n_lowercase = col_expr.str.count_matches(r"[a-z]").alias(f"{col}__n_lowercase")
                new_columns.append(n_lowercase)

            if "n_spaces" in self.features:
                n_spaces = col_expr.str.count_matches(r"\s").alias(f"{col}__n_spaces")
                new_columns.append(n_spaces)

            if "n_special" in self.features:
                # Special chars: not letters, digits, or spaces
                n_special = col_expr.str.count_matches(r"[^a-zA-Z0-9\s]").alias(f"{col}__n_special")
                new_columns.append(n_special)

            if "n_unique_chars" in self.features:
                # Count unique characters by exploding into chars and counting unique
                n_unique = col_expr.map_elements(
                    lambda x: len(set(x)) if x else 0, return_dtype=pl.Int64
                ).alias(f"{col}__n_unique_chars")
                new_columns.append(n_unique)

            # Ratio features
            str_len = col_expr.str.len_chars()

            if "ratio_uppercase" in self.features:
                n_upper = col_expr.str.count_matches(r"[A-Z]")
                ratio_upper = (
                    pl.when(str_len > 0).then(n_upper.cast(pl.Float64) / str_len).otherwise(0.0)
                ).alias(f"{col}__ratio_uppercase")
                new_columns.append(ratio_upper)

            if "ratio_digits" in self.features:
                n_dig = col_expr.str.count_matches(r"\d")
                ratio_dig = (
                    pl.when(str_len > 0).then(n_dig.cast(pl.Float64) / str_len).otherwise(0.0)
                ).alias(f"{col}__ratio_digits")
                new_columns.append(ratio_dig)

            if "ratio_special" in self.features:
                n_spec = col_expr.str.count_matches(r"[^a-zA-Z0-9\s]")
                ratio_spec = (
                    pl.when(str_len > 0).then(n_spec.cast(pl.Float64) / str_len).otherwise(0.0)
                ).alias(f"{col}__ratio_special")
                new_columns.append(ratio_spec)

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
