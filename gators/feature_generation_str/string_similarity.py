"""Fuzzy string similarity features between pairs of string columns."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import polars as pl
from pydantic import PrivateAttr, field_validator
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ..transformer._base_transformer import _BaseTransformer

_SIMILARITY_FUNCTIONS: dict[str, Callable[..., float]] = {
    "levenshtein": Levenshtein.normalized_similarity,
    "jaro_winkler": JaroWinkler.normalized_similarity,
}


class StringSimilarity(_BaseTransformer):
    """
    Generates fuzzy string similarity features between pairs of string columns.

    Computes a normalized similarity score (``0.0`` = completely different,
    ``1.0`` = identical) between each pair of columns using either Levenshtein
    (edit-distance based) or Jaro-Winkler similarity. This is particularly
    useful in fraud detection for identity matching, e.g. comparing the
    cardholder name against the shipping name, or the billing address against
    the shipping address.

    Parameters
    ----------
    subset_a : list[str]
        List of column names for the left side of each comparison.
    subset_b : list[str]
        List of column names for the right side of each comparison. Must have
        the same length as ``subset_a``; pairs are formed element-wise
        (``subset_a[i]`` compared to ``subset_b[i]``).
    method : Literal["levenshtein", "jaro_winkler"], default="levenshtein"
        Similarity algorithm to use:

        - ``'levenshtein'``: Normalized indel/edit-distance similarity.
        - ``'jaro_winkler'``: Jaro-Winkler similarity, which favors strings
          that share a common prefix (well suited to names).
    case_sensitive : bool, default=False
        Whether comparisons are case sensitive.
    new_column_names : list[str], optional
        Custom names for the similarity features. If ``None``, names are
        auto-generated as ``'{col_a}__{col_b}__{method}_similarity'``.
    drop_columns : bool, default=False
        Whether to drop the original ``subset_a``/``subset_b`` columns after
        creating the similarity features.

    Examples
    --------
    >>> from gators.feature_generation_str import StringSimilarity
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'billing_name': ['John Smith', 'Jane Doe'],
    ...     'shipping_name': ['Jon Smith', 'Robert Jones'],
    ... })

    **Example 1: Levenshtein similarity**

    >>> transformer = StringSimilarity(
    ...     subset_a=['billing_name'],
    ...     subset_b=['shipping_name'],
    ...     method='levenshtein',
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['billing_name', 'shipping_name', 'billing_name__shipping_name__levenshtein_similarity']

    **Example 2: Custom column names and drop_columns**

    >>> transformer = StringSimilarity(
    ...     subset_a=['billing_name'],
    ...     subset_b=['shipping_name'],
    ...     method='jaro_winkler',
    ...     new_column_names=['name_similarity'],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['name_similarity']
    """

    subset_a: list[str]
    subset_b: list[str]
    method: Literal["levenshtein", "jaro_winkler"] = "levenshtein"
    case_sensitive: bool = False
    new_column_names: list[str] | None = None
    drop_columns: bool = False
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @field_validator("subset_b", mode="after")
    @classmethod
    def check_lengths_match(cls, subset_b, info):
        subset_a = info.data.get("subset_a", [])
        if len(subset_a) != len(subset_b):
            raise ValueError(
                f"Length of subset_a ({len(subset_a)}) must match length of subset_b ({len(subset_b)})"
            )
        return subset_b

    @field_validator("new_column_names", mode="after")
    @classmethod
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            subset_a = info.data.get("subset_a", [])
            if len(new_column_names) != len(subset_a):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of subset_a ({len(subset_a)})"
                )
        return new_column_names

    @staticmethod
    def _default_name(col_a: str, col_b: str, method: str) -> str:
        return f"{col_a}__{col_b}__{method}_similarity"

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> StringSimilarity:
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        StringSimilarity
            Fitted transformer instance.
        """
        default_names = [
            self._default_name(a, b, self.method) for a, b in zip(self.subset_a, self.subset_b, strict=False)
        ]
        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = {d: [n] for d, n in zip(default_names, self.new_column_names, strict=False)}

        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating string similarity features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with similarity features appended.
        """
        sim_func: Callable[..., float] = _SIMILARITY_FUNCTIONS[self.method]
        case_sensitive = self.case_sensitive

        def _similarity(pair: dict) -> float | None:
            a, b = pair["a"], pair["b"]
            if a is None or b is None:
                return None
            if not case_sensitive:
                a, b = a.lower(), b.lower()
            return float(sim_func(a, b))

        new_columns = []
        for col_a, col_b in zip(self.subset_a, self.subset_b, strict=False):
            default_name = self._default_name(col_a, col_b, self.method)
            new_col_name = self._column_mapping[default_name][0]
            expr = (
                pl.struct([pl.col(col_a).alias("a"), pl.col(col_b).alias("b")])
                .map_elements(_similarity, return_dtype=pl.Float64)
                .alias(new_col_name)
            )
            new_columns.append(expr)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list(set(self.subset_a + self.subset_b))
            X = X.drop(columns_to_drop)

        return X
