from typing import Dict, List, Optional

import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer


class NGram(_BaseTransformer):
    """
    Extracts character or word n-grams from string columns.

    Creates count features for the most common n-grams, useful for tree-based
    models to capture local text patterns and sequences.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of string columns to extract n-grams from. If None, all string columns
        will be used.
    n : int, default=2
        Size of n-grams to extract (e.g., 2 for bigrams, 3 for trigrams).
    ngram_type : str, default="char"
        Type of n-grams to extract:

        - "char": Character-level n-grams (e.g., "ab", "bc" from "abc")
        - "word": Word-level n-grams (e.g., "hello world" as bigram)
    max_features : int, default=10
        Maximum number of most common n-grams to extract per column.
        Top-k n-grams are selected during fit() based on frequency.
    min_count : int, default=1
        Minimum number of occurrences for an n-gram to be included.
    drop_columns : bool, default=False
        Whether to drop the original string columns after feature extraction.

    Examples
    --------
    >>> from gators.feature_generation_str import NGram
    >>> import polars as pl

    >>> X =pl.DataFrame({
    ...     'text': ['hello world', 'hello there', 'world peace', None],
    ...     'desc': ['test data', 'test case', 'data case', '']
    ... })

    **Example 1: Character bigrams**

    >>> transformer = NGram(
    ...     subset=['text'],
    ...     n=2,
    ...     ngram_type='char',
    ...     max_features=5
    ... )
    >>> result = transformer.fit_transform(X)
    >>> print(result.columns)
    ['text', 'desc', 'text__ng_he', 'text__ng_el', 'text__ng_ll', 'text__ng_lo', 'text__ng_o_']
    # Features for top-5 character bigrams: 'he', 'el', 'll', 'lo', 'o '

    **Example 2: Word bigrams**

    >>> transformer = NGram(
    ...     subset=['text'],
    ...     n=2,
    ...     ngram_type='word',
    ...     max_features=3
    ... )
    >>> result = transformer.fit_transform(X)
    # Features for word bigrams: 'hello world', 'hello there', 'world peace'

    **Example 3: Character trigrams with min_count**

    >>> transformer = NGram(
    ...     subset=['desc'],
    ...     n=3,
    ...     ngram_type='char',
    ...     max_features=5,
    ...     min_count=2
    ... )
    >>> result = transformer.fit_transform(X)
    # Only trigrams appearing at least 2 times are considered

    **Example 4: Multiple columns with drop**

    >>> transformer = NGram(
    ...     subset=['text', 'desc'],
    ...     n=2,
    ...     ngram_type='char',
    ...     max_features=10,
    ...     drop_columns=True
    ... )
    >>> result = transformer.fit_transform(X)
    """

    subset: Optional[List[str]] = None
    n: int = 2
    ngram_type: str = "char"
    max_features: int = 10
    min_count: int = 1
    drop_columns: bool = False

    # Fitted attributes (not part of initialization)
    top_ngrams_: Dict[str, List[str]] = {}

    @field_validator("n")
    def check_n(cls, n):
        if n < 1:
            raise ValueError("n must be at least 1")
        if n > 10:
            raise ValueError("n should not exceed 10 for practical purposes")
        return n

    @field_validator("ngram_type")
    def check_ngram_type(cls, ngram_type):
        if ngram_type not in ["char", "word"]:
            raise ValueError("ngram_type must be 'char' or 'word'")
        return ngram_type

    @field_validator("max_features")
    def check_max_features(cls, max_features):
        if max_features < 1:
            raise ValueError("max_features must be at least 1")
        return max_features

    @field_validator("min_count")
    def check_min_count(cls, min_count):
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        return min_count

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "NGram":
        """Fit the transformer by identifying top-k n-grams for each column.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        NGram
            Fitted transformer instance.
        """
        if not self.subset:
            self.subset = [
                col
                for col, dtype in X.schema.items()
                if dtype == pl.String or dtype == pl.Utf8
            ]

        self.top_ngrams_ = {}

        for col in self.subset:
            # Get non-null values
            col_data = X.select(pl.col(col).fill_null("")).to_series()

            # Extract n-grams
            ngrams_list = []
            for text in col_data:
                text = str(text)
                if self.ngram_type == "char":
                    # Character n-grams
                    for i in range(len(text) - self.n + 1):
                        ngrams_list.append(text[i : i + self.n])
                else:
                    # Word n-grams
                    words = text.split()
                    for i in range(len(words) - self.n + 1):
                        ngram = " ".join(words[i : i + self.n])
                        ngrams_list.append(ngram)

            # Count n-grams
            if ngrams_list:
                ngram_counts = (
                    pl.DataFrame({"ngram": ngrams_list})
                    .group_by("ngram")
                    .agg(pl.len().alias("count"))
                )

                # Filter by min_count and get top-k
                top_ngrams = (
                    ngram_counts.filter(pl.col("count") >= self.min_count)
                    .sort(
                        ["count", "ngram"], descending=[True, False]
                    )  # Sort by count desc, then ngram asc for consistency
                    .head(self.max_features)
                    .select("ngram")
                    .to_series()
                    .to_list()
                )

                self.top_ngrams_[col] = top_ngrams
            else:
                self.top_ngrams_[col] = []

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating n-gram count features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with n-gram count features.
        """
        if self.subset is None:
            return X
            
        new_columns = []

        for col in self.subset:
            if col not in self.top_ngrams_ or not self.top_ngrams_[col]:
                continue

            col_expr = pl.col(col).fill_null("")

            for ngram in self.top_ngrams_[col]:
                # Create safe feature name by replacing special chars
                safe_ngram = (
                    ngram.replace(" ", "_")
                    .replace(".", "dot")
                    .replace(",", "comma")
                    .replace("!", "excl")
                    .replace("?", "ques")
                    .replace("#", "hash")
                    .replace("@", "at")
                    .replace("-", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace("'", "")
                    .replace('"', "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("{", "")
                    .replace("}", "")
                    .replace("|", "")
                    .replace("&", "and")
                    .replace("+", "plus")
                    .replace("=", "eq")
                    .replace("<", "lt")
                    .replace(">", "gt")
                    .replace(":", "")
                    .replace(";", "")
                )

                # Truncate if too long (keep first 20 chars)
                if len(safe_ngram) > 20:
                    safe_ngram = safe_ngram[:20]

                # Count occurrences using overlapping matches
                # map_elements allows overlapping count by incrementing position by 1
                def count_overlapping(text, pattern):
                    if not text or not pattern:
                        return 0
                    count = 0
                    start = 0
                    while True:
                        pos = text.find(pattern, start)
                        if pos == -1:
                            break
                        count += 1
                        start = pos + 1  # Move by 1 to allow overlapping
                    return count

                count_expr = col_expr.map_elements(
                    lambda x: count_overlapping(str(x), ngram), return_dtype=pl.Int64
                ).alias(f"{col}__ng_{safe_ngram}")
                new_columns.append(count_expr)

        X = X.with_columns(new_columns)

        if self.drop_columns and self.subset is not None:
            X = X.drop(self.subset)

        return X
