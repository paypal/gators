"""Word-level statistical features for string columns."""

from __future__ import annotations

import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

_ALL_FEATURES_ORDER = [
    "n_words",
    "avg_word_length",
    "n_unique_words",
    "max_word_length",
    "min_word_length",
]


class WordStatistics(_BaseTransformer):
    """
    Generates word-level statistical features from string columns.

    Tokenizes each string on whitespace and computes summary statistics over
    the resulting words, which are useful for tree-based models to identify
    patterns in free-text fields (e.g. addresses, descriptions, names).

    Parameters
    ----------
    subset : list[str], default=None
        List of string columns to extract features from. If None, all string
        columns will be used.
    features : list[str], default=["n_words", "avg_word_length", "n_unique_words", "max_word_length", "min_word_length"]
        Word statistics to generate. Options:

        - "n_words": Number of whitespace-separated words
        - "avg_word_length": Average number of characters per word
        - "n_unique_words": Number of distinct words
        - "max_word_length": Length of the longest word
        - "min_word_length": Length of the shortest word
    drop_columns : bool, default=False
        Whether to drop the original string columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_str import WordStatistics
    >>> import polars as pl

    >>> X = pl.DataFrame({'address': ['123 Main St', 'One Infinite Loop', '', None]})

    **Example 1: Default features**

    >>> transformer = WordStatistics(subset=['address'])
    >>> result = transformer.fit_transform(X)
    >>> result['address__n_words'].to_list()
    [3.0, 3.0, 0.0, 0.0]

    **Example 2: Selected features with drop_columns**

    >>> transformer = WordStatistics(
    ...     subset=['address'],
    ...     features=['n_words', 'avg_word_length'],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['address__n_words', 'address__avg_word_length']
    """

    subset: list[str] | None = None
    features: list[str] = [
        "n_words",
        "avg_word_length",
        "n_unique_words",
        "max_word_length",
        "min_word_length",
    ]
    drop_columns: bool = False
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @field_validator("features")
    def check_features(cls, features):
        for feature in features:
            if feature not in _ALL_FEATURES_ORDER:
                raise ValueError(
                    f"Feature '{feature}' is not supported. "
                    f"Supported features: {_ALL_FEATURES_ORDER}"
                )
        return features

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> WordStatistics:
        """Fit the transformer by identifying string columns if not specified.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        WordStatistics
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col for col, dtype in X.schema.items() if dtype == pl.String or dtype == pl.Utf8
            ]
        ordered_features = [f for f in _ALL_FEATURES_ORDER if f in self.features]
        self._column_mapping = {
            col: [f"{col}__{f}" for f in ordered_features] for col in self.subset
        }
        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating word statistics features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with word statistics features.
        """
        if self.subset is None:
            return X  # pragma: no cover

        new_columns = []

        for col in self.subset:
            words = pl.col(col).fill_null("").str.extract_all(r"\S+")
            n_words = words.list.len().cast(pl.Float64)
            word_lengths = words.list.eval(pl.element().str.len_chars())

            if "n_words" in self.features:
                new_columns.append(n_words.alias(f"{col}__n_words"))

            if "avg_word_length" in self.features:
                avg_word_length = (
                    pl.when(n_words > 0)
                    .then(word_lengths.list.mean())
                    .otherwise(0.0)
                    .alias(f"{col}__avg_word_length")
                )
                new_columns.append(avg_word_length)

            if "n_unique_words" in self.features:
                n_unique_words = (
                    words.list.n_unique().cast(pl.Float64).alias(f"{col}__n_unique_words")
                )
                new_columns.append(n_unique_words)

            if "max_word_length" in self.features:
                max_word_length = (
                    pl.when(n_words > 0)
                    .then(word_lengths.list.max().cast(pl.Float64))
                    .otherwise(0.0)
                    .alias(f"{col}__max_word_length")
                )
                new_columns.append(max_word_length)

            if "min_word_length" in self.features:
                min_word_length = (
                    pl.when(n_words > 0)
                    .then(word_lengths.list.min().cast(pl.Float64))
                    .otherwise(0.0)
                    .alias(f"{col}__min_word_length")
                )
                new_columns.append(min_word_length)

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
