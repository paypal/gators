"""TF-IDF feature generator for string columns."""

from __future__ import annotations

import math
import re

import polars as pl
from pydantic import PositiveInt, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer


class TfidfFeatures(_BaseTransformer):
    """Compute TF-IDF features for string columns.

    For each column in ``subset`` and each token in the learned vocabulary,
    a new column ``'{col}__tfidf_{token}'`` is appended whose values equal:

    .. math::

        \\text{TF-IDF}(t, d) = \\frac{\\text{count}(t, d)}{|d|} \\times
        \\left(\\ln\\frac{1 + N}{1 + \\text{df}(t)} + 1\\right)

    where :math:`|d|` is the number of tokens in document *d*, *N* is the
    total number of training documents, and df(t) is the document frequency
    of token *t*.

    Algorithm (fit)
    ---------------
    1. Tokenise each non-null document by splitting on ``separator``.
    2. Optionally lower-case tokens (``lowercase=True``).
    3. Count the document frequency (df) for each unique token.
    4. Discard tokens that appear in fewer than ``min_df`` documents.
    5. Select the top ``max_features`` tokens by df (ties broken by order).
    6. Store the IDF weight for each selected token.

    Algorithm (transform)
    ---------------------
    For every selected token, count its occurrences in each document, divide
    by the document length (TF normalisation), and multiply by the stored IDF.
    Documents that are ``null`` or empty produce ``0.0`` for all tokens.

    Parameters
    ----------
    subset : list[str] or None, default=None
        String columns to vectorise.  When ``None``, all ``String`` columns
        are selected automatically during ``fit``.
    max_features : PositiveInt, default=100
        Maximum number of tokens (features) per column.
    separator : str, default=' '
        Token delimiter used to split text into tokens.
    min_df : PositiveInt, default=1
        Minimum number of documents a token must appear in to be retained.
    lowercase : bool, default=True
        Lower-case text before tokenisation.
    drop_columns : bool, default=False
        Drop the original string columns after appending TF-IDF features.

    Attributes
    ----------
    vocabulary_ : dict[str, list[str]]
        Mapping of column name to its ordered vocabulary list
        (set after ``fit``).
    idf_ : dict[str, dict[str, float]]
        Mapping of column name to ``{token: idf_weight}``
        (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation_str import TfidfFeatures

    >>> X = pl.DataFrame({
    ...     "text": [
    ...         "the cat sat on the mat",
    ...         "the dog sat on the log",
    ...         "the cat and the dog",
    ...     ]
    ... })
    >>> transformer = TfidfFeatures(subset=["text"], max_features=5)
    >>> transformer.fit(X)
    TfidfFeatures(subset=['text'], max_features=5, separator=' ', min_df=1, lowercase=True, drop_columns=False)
    >>> result = transformer.transform(X)
    >>> [c for c in result.columns if c.startswith("text__tfidf_")]
    ['text__tfidf_the', 'text__tfidf_sat', 'text__tfidf_on', 'text__tfidf_cat', 'text__tfidf_dog']
    """

    subset: list[str] | None = None
    max_features: PositiveInt = 100
    separator: str = " "
    min_df: PositiveInt = 1
    lowercase: bool = True
    drop_columns: bool = False

    _vocabulary: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _idf: dict[str, dict[str, float]] = PrivateAttr(default_factory=dict)
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @property
    def vocabulary_(self) -> dict[str, list[str]]:
        """Learned vocabulary per column."""
        return self._vocabulary

    @property
    def idf_(self) -> dict[str, dict[str, float]]:
        """IDF weights per column and token."""
        return self._idf

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> TfidfFeatures:
        """Learn the vocabulary and IDF weights from *X*.

        Parameters
        ----------
        X : pl.DataFrame
            Training DataFrame.
        y : pl.Series or None, default=None
            Ignored; present for sklearn compatibility.

        Returns
        -------
        TfidfFeatures
            The fitted transformer instance.
        """
        if self.subset is None:
            self.subset = [col for col, dt in zip(X.columns, X.dtypes, strict=False) if dt == pl.String]

        for col in self.subset:
            docs = X[col].drop_nulls()
            if self.lowercase:
                docs = docs.str.to_lowercase()
            n_docs = len(docs)

            df_col = pl.DataFrame({col: docs})
            df_col = df_col.with_columns(pl.col(col).str.split(self.separator).alias("tokens"))
            # doc_id lets us compute per-document unique tokens
            df_col = df_col.with_row_index("__doc_id__")
            df_exploded = df_col.explode("tokens", empty_as_null=True).filter(
                pl.col("tokens").str.len_chars() > 0
            )
            # Unique (doc_id, token) pairs → document frequency
            df_unique = df_exploded.select(["__doc_id__", "tokens"]).unique()
            df_freq = (
                df_unique.group_by("tokens")
                .agg(pl.len().alias("df"))
                .filter(pl.col("df") >= self.min_df)
                .sort("df", descending=True)
                .head(self.max_features)
            )
            vocabulary = df_freq["tokens"].to_list()
            self._vocabulary[col] = vocabulary

            df_values = df_freq["df"].to_list()
            self._idf[col] = {
                token: math.log((1 + n_docs) / (1 + int(df_val))) + 1.0
                for token, df_val in zip(vocabulary, df_values, strict=False)
            }

        self._column_mapping = {
            col: [f"{col}__tfidf_{token}" for token in vocab]
            for col, vocab in self._vocabulary.items()
        }
        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Append TF-IDF feature columns to *X*.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with TF-IDF columns appended.
        """
        if self.subset is None:
            return X  # pragma: no cover

        result = X

        # Step 1 — add one token-list column per source column
        token_list_exprs = []
        temp_col_names: dict[str, str] = {}
        for col in self.subset:
            if col not in self._vocabulary:
                continue
            source = pl.col(col).str.to_lowercase() if self.lowercase else pl.col(col)
            temp = f"__tokens_{col}__"
            temp_col_names[col] = temp
            token_list_exprs.append(
                source.str.split(self.separator)
                .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
                .alias(temp)
            )

        if token_list_exprs:
            result = result.with_columns(token_list_exprs)

        # Step 2 — one TF-IDF expression per (column, token)
        tfidf_exprs = []
        for col in self.subset:
            if col not in self._vocabulary or col not in temp_col_names:
                continue
            temp = temp_col_names[col]
            for token in self._vocabulary[col]:
                idf = self._idf[col][token]
                safe = re.sub(r"\W+", "_", token).strip("_")
                new_col = f"{col}__tfidf_{safe}"

                count_expr = pl.col(temp).list.count_matches(token).fill_null(0).cast(pl.Float64)
                total_expr = pl.col(temp).list.len().fill_null(0).cast(pl.Float64)
                tfidf_exprs.append(
                    pl.when(total_expr > 0)
                    .then(count_expr / total_expr * idf)
                    .otherwise(0.0)
                    .alias(new_col)
                )

        if tfidf_exprs:
            result = result.with_columns(tfidf_exprs)

        # Drop temporary token-list columns
        if temp_col_names:
            result = result.drop(list(temp_col_names.values()))

        if self.drop_columns:
            result = result.drop([c for c in self.subset if c in result.columns])

        return result
