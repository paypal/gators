"""Named-group regex extraction as new columns."""

from __future__ import annotations

import re

import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer


class RegexExtractFeatures(_BaseTransformer):
    """Extract named capture groups from string columns as new features.

    For each source column in ``subset``, applies the regex ``pattern`` and
    creates one new column per named group following the naming convention
    ``'{source_col}__{group_name}'``.  Rows that do not match produce
    ``null`` in the extracted columns.

    Parameters
    ----------
    subset : list[str]
        String columns to apply the regex to.
    pattern : str
        Regular expression containing at least one named capture group,
        e.g. ``r'(?P<area>\\d{3})-(?P<number>\\d{7})'``.
    drop_columns : bool, default=False
        When ``True``, drop the original ``subset`` columns after extraction.

    Attributes
    ----------
    group_names_ : list[str]
        Named groups discovered in ``pattern`` (set after ``fit``).

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_generation_str import RegexExtractFeatures

    >>> X = pl.DataFrame({
    ...     "phone": ["123-4567890", "987-1234567", None],
    ...     "code":  ["AB-001", "CD-999", "EF-123"],
    ... })
    >>> transformer = RegexExtractFeatures(
    ...     subset=["phone", "code"],
    ...     pattern=r"(?P<prefix>\\w+)-(?P<suffix>\\w+)",
    ... )
    >>> transformer.fit(X)
    RegexExtractFeatures(subset=['phone', 'code'], pattern='(?P<prefix>\\\\w+)-(?P<suffix>\\\\w+)', drop_columns=False)
    >>> transformer.transform(X)
    shape: (3, 6)
    ┌─────────────┬────────┬───────────────┬──────────────┬──────────────┬─────────────┐
    │ phone       ┆ code   ┆ phone__prefix ┆ phone__suffix┆ code__prefix ┆ code__suffix│
    │ ---         ┆ ---    ┆ ---           ┆ ---          ┆ ---          ┆ ---         │
    │ str         ┆ str    ┆ str           ┆ str          ┆ str          ┆ str         │
    ╞═════════════╪════════╪═══════════════╪══════════════╪══════════════╪═════════════╡
    │ 123-4567890 ┆ AB-001 ┆ 123           ┆ 4567890      ┆ AB           ┆ 001         │
    │ 987-1234567 ┆ CD-999 ┆ 987           ┆ 1234567      ┆ CD           ┆ 999         │
    │ null        ┆ EF-123 ┆ null          ┆ null         ┆ EF           ┆ 123         │
    └─────────────┴────────┴───────────────┴──────────────┴──────────────┴─────────────┘
    """

    subset: list[str]
    pattern: str
    drop_columns: bool = False

    _group_names: list[str] = PrivateAttr(default_factory=list)
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str) -> str:
        groups = re.findall(r"\(\?P<([^>]+)>", v)
        if not groups:
            raise ValueError(
                "pattern must contain at least one named capture group, "
                "e.g. r'(?P<name>\\w+)'.  No named groups found."
            )
        return v

    @property
    def group_names_(self) -> list[str]:
        """Named groups discovered in the pattern."""
        return self._group_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> RegexExtractFeatures:
        """Parse group names from ``pattern``.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame (not used for computation).
        y : pl.Series or None, default=None
            Ignored; present for sklearn compatibility.

        Returns
        -------
        RegexExtractFeatures
            The fitted transformer instance.
        """
        self._group_names = re.findall(r"\(\?P<([^>]+)>", self.pattern)
        self._column_mapping = {
            col: [f"{col}__{group}" for group in self._group_names] for col in self.subset
        }
        self._output_dtypes = {
            new: pl.String for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Extract named groups and append them as new columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with one new column per named group per source column.
        """
        result = X
        for col in self.subset:
            tmp = f"__regex_tmp_{col}__"
            result = result.with_columns(pl.col(col).str.extract_groups(self.pattern).alias(tmp))
            result = result.unnest(tmp)
            rename_map = {group: f"{col}__{group}" for group in self._group_names}
            result = result.rename(rename_map)

        if self.drop_columns:
            result = result.drop(self.subset)
        return result
