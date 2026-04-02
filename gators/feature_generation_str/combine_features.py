from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel, field_validator
from sklearn.base import BaseEstimator, TransformerMixin


class CombineFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Combines specific string/categorical columns to create composite key features (UID-like).

    Unlike InteractionFeatures which generates all combinations, this transformer
    only creates the specific column combinations you provide, making it more
    efficient for creating unique identifiers or specific interaction features.

    Parameters
    ----------
    column_groups : List[List[str]]
        List of column groups to combine. Each group is a list of column names
        that will be concatenated together.
        Example: [['cat1', 'cat2'], ['cat1', 'addr1']]
    separator : str, default='_'
        String to use as separator when combining column values.
    drop_columns : bool, default=False
        Whether to drop the original columns after creating combinations.
    new_column_names : Optional[List[str]], default=None
        List of custom names for the combined columns. If None, uses default naming
        pattern where columns are joined with '__' (e.g., 'cat1__cat2').
        Must have same length as column_groups.

    Examples
    --------
    >>> from gators.feature_generation_str import CombineFeatures
    >>> import polars as pl

    >>> X ={
    ...     'cat1': ['A', 'A', 'B', 'B', 'A'],
    ...     'cat2': ['X', 'Y', 'X', 'Y', 'X'],
    ...     'addr1': ['US', 'US', 'UK', 'UK', 'CA'],
    ...     'amount': [100, 200, 150, 300, 250]
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Basic combination**

    >>> transformer = CombineFeatures(
    ...     column_groups=[['cat1', 'cat2']]
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (5, 5)
    ┌───────┬───────┬───────┬────────┬──────────────┐
    │ cat1 ┆ cat2 ┆ addr1 ┆ amount ┆ cat1__cat2 │
    │ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---          │
    │ str   ┆ str   ┆ str   ┆ i64    ┆ str          │
    ╞═══════╪═══════╪═══════╪════════╪══════════════╡
    │ A     ┆ X     ┆ US    ┆ 100    ┆ A_X          │
    │ A     ┆ Y     ┆ US    ┆ 200    ┆ A_Y          │
    │ B     ┆ X     ┆ UK    ┆ 150    ┆ B_X          │
    │ B     ┆ Y     ┆ UK    ┆ 300    ┆ B_Y          │
    │ A     ┆ X     ┆ CA    ┆ 250    ┆ A_X          │
    └───────┴───────┴───────┴────────┴──────────────┘

    **Example 2: Multiple combinations**

    >>> transformer = CombineFeatures(
    ...     column_groups=[['cat1', 'cat2'], ['cat1', 'addr1']]
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (5, 6)
    ┌───────┬───────┬───────┬────────┬──────────────┬───────────────┐
    │ cat1 ┆ cat2 ┆ addr1 ┆ amount ┆ cat1__cat2 ┆ cat1__addr1  │
    │ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---          ┆ ---           │
    │ str   ┆ str   ┆ str   ┆ i64    ┆ str          ┆ str           │
    ╞═══════╪═══════╪═══════╪════════╪══════════════╪═══════════════╡
    │ A     ┆ X     ┆ US    ┆ 100    ┆ A_X          ┆ A_US          │
    │ A     ┆ Y     ┆ US    ┆ 200    ┆ A_Y          ┆ A_US          │
    │ B     ┆ X     ┆ UK    ┆ 150    ┆ B_X          ┆ B_UK          │
    │ B     ┆ Y     ┆ UK    ┆ 300    ┆ B_Y          ┆ B_UK          │
    │ A     ┆ X     ┆ CA    ┆ 250    ┆ A_X          ┆ A_CA          │
    └───────┴───────┴───────┴────────┴──────────────┴───────────────┘

    **Example 3: Custom separator and column names**

    >>> transformer = CombineFeatures(
    ...     column_groups=[['cat1', 'cat2', 'addr1']],
    ...     separator='|',
    ...     new_column_names=['uid']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (5, 5)
    ┌───────┬───────┬───────┬────────┬──────────┐
    │ cat1 ┆ cat2 ┆ addr1 ┆ amount ┆ uid      │
    │ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---      │
    │ str   ┆ str   ┆ str   ┆ i64    ┆ str      │
    ╞═══════╪═══════╪═══════╪════════╪══════════╡
    │ A     ┆ X     ┆ US    ┆ 100    ┆ A|X|US   │
    │ A     ┆ Y     ┆ US    ┆ 200    ┆ A|Y|US   │
    │ B     ┆ X     ┆ UK    ┆ 150    ┆ B|X|UK   │
    │ B     ┆ Y     ┆ UK    ┆ 300    ┆ B|Y|UK   │
    │ A     ┆ X     ┆ CA    ┆ 250    ┆ A|X|CA   │
    └───────┴───────┴───────┴────────┴──────────┘

    **Example 4: Creating UIDs for fraud detection**

    >>> # Create composite keys for unique user identification
    >>> transformer = CombineFeatures(
    ...     column_groups=[
    ...         ['cat1', 'cat2', 'card3'],  # Card combination
    ...         ['cat1', 'addr1'],            # Card + address
    ...         ['email_domain', 'cat1']      # Email + card
    ...     ],
    ...     new_column_names=['card_uid', 'card_addr_uid', 'email_card_uid']
    ... )
    >>> # Can then use these UIDs for frequency encoding or groupby operations

    Notes
    -----

    - Null values are converted to string "null" before concatenation
    - All column values are cast to string before combining
    - Useful for creating unique identifiers (UIDs) for user/card tracking
    - More efficient than InteractionFeatures when you know exactly which
      combinations you need
    """

    column_groups: List[List[str]]
    separator: str = "_"
    drop_columns: bool = False
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = {}

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            column_groups = info.data.get("column_groups", [])
            if len(new_column_names) != len(column_groups):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of column_groups ({len(column_groups)})"
                )
        return new_column_names

    def fit(
        self, X: pl.DataFrame, y: Optional[pl.Series] = None
    ) -> "CombineFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        CombineFeatures
            Fitted transformer instance.
        """
        default_names = ["__".join(group) for group in self.column_groups]

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by combining categorical columns.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with combined categorical features.
        """
        new_columns = []
        columns_to_drop = set()

        for group in self.column_groups:
            default_name = "__".join(group)
            new_col_name = self._column_mapping[default_name]

            # Concatenate columns with separator
            # Cast to string and handle nulls
            concat_expr = pl.concat_str(
                [pl.col(col).cast(pl.Utf8).fill_null("null") for col in group],
                separator=self.separator,
            ).alias(new_col_name)

            new_columns.append(concat_expr)

            if self.drop_columns:
                columns_to_drop.update(group)

        X = X.with_columns(new_columns)

        if self.drop_columns and columns_to_drop:
            X = X.drop(list(columns_to_drop))

        return X
