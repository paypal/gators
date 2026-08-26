import polars as pl
from pydantic import PositiveFloat, field_validator

from ..transformer._base_transformer import _BaseTransformer


class HHIFeatures(_BaseTransformer):
    """
    Generates Herfindahl–Hirschman Index (HHI) features from groups of columns.

    For each group of columns, the HHI is computed as:

    .. math::

        HHI = \\sum_{i=1}^{n} s_i^2, \\quad s_i = \\frac{x_i}{\\sum_j x_j + \\varepsilon}

    where :math:`s_i` is the row-wise market share of column :math:`x_i` within its
    group, and :math:`\\varepsilon` is a small constant added to the total for
    numerical stability.

    The HHI ranges from :math:`1/n` (perfect equality, minimum concentration) to
    ``1.0`` (total concentration, one column dominates).  A value near ``1.0``
    signals that a single component accounts for virtually all of the group total,
    whereas a value near :math:`1/n` indicates a uniform distribution across the
    :math:`n` columns.

    Parameters
    ----------
    column_groups : list[list[str]]
        One inner list of column names per output feature.  Each inner list must
        contain at least two column names.  One HHI value is produced per group.
    epsilon : float, optional
        Small positive constant added to the total sum to prevent division by zero,
        by default ``1e-8``.
    new_column_names : list[str], optional
        Custom output column names.  If ``None``, names are auto-generated as
        ``'{col1}__{col2}__...__hhi'``.
    drop_columns : bool, optional
        Whether to drop all source columns after creating the HHI features,
        by default ``False``.

    Examples
    --------
    >>> from gators.feature_generation import HHIFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'brand_a': [10.0, 20.0, 0.0],
    ...     'brand_b': [30.0, 10.0, 0.0],
    ...     'brand_c': [60.0, 70.0, 0.0],
    ... })

    **Example 1: Single group**

    >>> transformer = HHIFeatures(
    ...     column_groups=[['brand_a', 'brand_b', 'brand_c']],
    ... )
    >>> transformer.fit(X)
    HHIFeatures(column_groups=[['brand_a', 'brand_b', 'brand_c']])
    >>> result = transformer.transform(X)
    >>> 'brand_a__brand_b__brand_c__hhi' in result.columns
    True

    **Example 2: Multiple groups**

    Compute separate HHI values for two independent column groups.

    >>> X2 = pl.DataFrame({
    ...     'a1': [10.0, 50.0],
    ...     'a2': [90.0, 50.0],
    ...     'b1': [25.0, 25.0],
    ...     'b2': [25.0, 75.0],
    ... })
    >>> transformer = HHIFeatures(
    ...     column_groups=[['a1', 'a2'], ['b1', 'b2']],
    ... )
    >>> result = transformer.fit_transform(X2)
    >>> 'a1__a2__hhi' in result.columns
    True
    >>> 'b1__b2__hhi' in result.columns
    True

    **Example 3: Custom column names**

    >>> transformer = HHIFeatures(
    ...     column_groups=[['brand_a', 'brand_b', 'brand_c']],
    ...     new_column_names=['market_hhi'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'market_hhi' in result.columns
    True

    **Example 4: With drop_columns=True**

    >>> transformer = HHIFeatures(
    ...     column_groups=[['brand_a', 'brand_b']],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'brand_a' in result.columns
    False
    >>> 'brand_b' in result.columns
    False
    >>> 'brand_c' in result.columns
    True
    """

    column_groups: list[list[str]]
    epsilon: PositiveFloat = 1e-8
    new_column_names: list[str] | None = None
    drop_columns: bool = False

    @field_validator("column_groups", mode="after")
    @classmethod
    def check_groups_non_empty(cls, column_groups):
        for i, group in enumerate(column_groups):
            if len(group) < 2:
                raise ValueError(f"column_groups[{i}] must contain at least two column names.")
        return column_groups

    @field_validator("new_column_names", mode="after")
    @classmethod
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            column_groups = info.data.get("column_groups", [])
            if len(new_column_names) != len(column_groups):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of column_groups ({len(column_groups)})"
                )
        return new_column_names

    @staticmethod
    def _default_name(cols: list[str]) -> str:
        return "__".join(cols) + "__hhi"

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "HHIFeatures":
        """Fit the transformer by resolving column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        HHIFeatures
            Fitted transformer instance.
        """
        if self.new_column_names is None:
            self.new_column_names = [self._default_name(group) for group in self.column_groups]

        self._output_dtypes = dict.fromkeys(self.new_column_names, pl.Float64)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating HHI features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with HHI features appended.
        """
        new_columns = []

        assert self.new_column_names is not None
        for group, new_col_name in zip(self.column_groups, self.new_column_names, strict=False):
            float_cols = [pl.col(c).cast(pl.Float64) for c in group]
            total = pl.sum_horizontal(float_cols) + self.epsilon
            squared_shares = [(pl.col(c).cast(pl.Float64) / total) ** 2 for c in group]
            new_columns.append(pl.sum_horizontal(squared_shares).alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list({col for group in self.column_groups for col in group})
            X = X.drop(columns_to_drop)

        return X
