import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer


class ConcentrationIndexFeatures(_BaseTransformer):
    """
    Generates concentration index features by dividing a numerator column by the
    row-wise sum of a group of denominator columns.

    Each feature is computed as:

        ``numerator / (denom_1 + denom_2 + ... + denom_n [+ 1])``

    The optional ``+1`` (Laplace smoothing, enabled via ``smoothing=True``) prevents
    division-by-zero when all denominator columns are zero for a given row. This
    is useful for computing market share, click-through concentration, or any ratio
    of one quantity against the total of several others.

    Parameters
    ----------
    numerator_columns : list[str]
        List of column names to use as numerators. One concentration index is
        created per entry.
    denominator_columns : list[list[str]]
        List of denominator column groups. Each inner list corresponds to the
        denominator columns for the matching numerator. Must have the same length
        as ``numerator_columns``, and each inner list must be non-empty.
    smoothing : bool, optional
        Whether to add ``1`` to each denominator sum to prevent division by zero,
        by default ``True``.
    new_column_names : list[str], optional
        Custom output column names. If ``None``, names are auto-generated as
        ``'{numerator}__conc__{denom1}__{denom2}__...'``, by default ``None``.
    drop_columns : bool, optional
        Whether to drop the original numerator and denominator columns after
        creating the concentration features, by default ``False``.

    Examples
    --------
    >>> from gators.feature_generation import ConcentrationIndexFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'brand_a': [10, 20, 0, 5],
    ...     'brand_b': [30, 10, 0, 15],
    ...     'brand_c': [60, 70, 0, 80],
    ... })

    **Example 1: Single concentration index**

    Compute the share of ``brand_a`` within the total of ``brand_b`` and ``brand_c``.

    >>> transformer = ConcentrationIndexFeatures(
    ...     numerator_columns=['brand_a'],
    ...     denominator_columns=[['brand_b', 'brand_c']],
    ... )
    >>> transformer.fit(X)
    ConcentrationIndexFeatures(numerator_columns=['brand_a'], denominator_columns=[['brand_b', 'brand_c']], smoothing=True)
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 4)
    ┌─────────┬─────────┬─────────┬───────────────────────────────┐
    │ brand_a │ brand_b │ brand_c │ brand_a__conc__brand_b__brand… │
    │ i64     │ i64     │ i64     │ f64                            │
    ├─────────┼─────────┼─────────┼───────────────────────────────┤
    │ 10      │ 30      │ 60      │ 0.109890                       │
    │ 20      │ 10      │ 70      │ 0.246914                       │
    │ 0       │ 0       │ 0       │ 0.0                            │
    │ 5       │ 15      │ 80      │ 0.052083                       │
    └─────────┴─────────┴─────────┴───────────────────────────────┘

    **Example 2: Multiple concentration indices**

    Each numerator has its own denominator group.

    >>> X2 = pl.DataFrame({
    ...     'clicks_a': [10, 20, 0],
    ...     'clicks_b': [5, 15, 10],
    ...     'views_x': [100, 200, 0],
    ...     'views_y': [50, 100, 0],
    ...     'impressions_x': [200, 400, 500],
    ...     'impressions_y': [300, 600, 500],
    ... })
    >>> transformer = ConcentrationIndexFeatures(
    ...     numerator_columns=['clicks_a', 'clicks_b'],
    ...     denominator_columns=[
    ...         ['views_x', 'views_y'],
    ...         ['impressions_x', 'impressions_y'],
    ...     ],
    ... )
    >>> result = transformer.fit_transform(X2)

    **Example 3: No smoothing**

    >>> X3 = pl.DataFrame({
    ...     'sales': [10, 20, 0],
    ...     'total_a': [5, 10, 0],
    ...     'total_b': [5, 10, 0],
    ... })
    >>> transformer = ConcentrationIndexFeatures(
    ...     numerator_columns=['sales'],
    ...     denominator_columns=[['total_a', 'total_b']],
    ...     smoothing=False,
    ... )
    >>> result = transformer.fit_transform(X3)
    >>> result
    shape: (3, 4)
    ┌───────┬─────────┬─────────┬────────────────────────────────────┐
    │ sales │ total_a │ total_b │ sales__conc__total_a__total_b      │
    │ i64   │ i64     │ i64     │ f64                                │
    ├───────┼─────────┼─────────┼────────────────────────────────────┤
    │ 10    │ 5       │ 5       │ 1.0                                │
    │ 20    │ 10      │ 10      │ 1.0                                │
    │ 0     │ 0       │ 0       │ NaN                                │
    └───────┴─────────┴─────────┴────────────────────────────────────┘

    **Example 4: Custom column names**

    >>> transformer = ConcentrationIndexFeatures(
    ...     numerator_columns=['brand_a'],
    ...     denominator_columns=[['brand_b', 'brand_c']],
    ...     new_column_names=['brand_a_share'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 4)
    ┌─────────┬─────────┬─────────┬───────────────┐
    │ brand_a │ brand_b │ brand_c │ brand_a_share │
    │ i64     │ i64     │ i64     │ f64           │
    ├─────────┼─────────┼─────────┼───────────────┤
    │ 10      │ 30      │ 60      │ 0.109890      │
    │ 20      │ 10      │ 70      │ 0.246914      │
    │ 0       │ 0       │ 0       │ 0.0           │
    │ 5       │ 15      │ 80      │ 0.052083      │
    └─────────┴─────────┴─────────┴───────────────┘

    **Example 5: With drop_columns=True**

    >>> transformer = ConcentrationIndexFeatures(
    ...     numerator_columns=['brand_a'],
    ...     denominator_columns=[['brand_b', 'brand_c']],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (4, 2)
    ┌─────────┬───────────────────────────────┐
    │ brand_b │ brand_a__conc__brand_b__brand… │
    │ i64     │ f64                            │
    ├─────────┼───────────────────────────────┤
    ...
    └─────────┴───────────────────────────────┘

    **Example 6: Null behavior**

    Nulls in the *numerator* propagate to the result. Nulls in denominator
    columns are treated as ``0`` by ``pl.sum_horizontal`` (they are ignored),
    so a row where all denominators are ``null`` behaves as if the denominator
    sum were ``0`` (and becomes ``1`` with smoothing enabled).

    >>> X_nulls = pl.DataFrame({
    ...     'A': [10, None, 30],
    ...     'B': [5, 5, None],
    ...     'C': [5, 5, None],
    ... })
    >>> transformer = ConcentrationIndexFeatures(
    ...     numerator_columns=['A'],
    ...     denominator_columns=[['B', 'C']],
    ... )
    >>> result = transformer.fit_transform(X_nulls)
    >>> result
    shape: (3, 4)
    ┌──────┬──────┬──────┬───────────────┐
    │ A    │ B    │ C    │ A__conc__B__C │
    │ i64  │ i64  │ i64  │ f64           │
    ├──────┼──────┼──────┼───────────────┤
    │ 10   │ 5    │ 5    │ 0.909091      │
    │ null │ 5    │ 5    │ null          │
    │ 30   │ null │ null │ 30.0          │
    └──────┴──────┴──────┴───────────────┘
    """

    numerator_columns: list[str]
    denominator_columns: list[list[str]]
    smoothing: bool = True
    new_column_names: list[str] | None = None
    drop_columns: bool = False
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @field_validator("denominator_columns", mode="after")
    def check_lengths_match(cls, denominator_columns, info):
        numerator_columns = info.data.get("numerator_columns", [])
        if len(numerator_columns) != len(denominator_columns):
            raise ValueError(
                f"Length of numerator_columns ({len(numerator_columns)}) "
                f"must match length of denominator_columns ({len(denominator_columns)})"
            )
        for i, group in enumerate(denominator_columns):
            if len(group) == 0:
                raise ValueError(f"denominator_columns[{i}] must contain at least one column name.")
        return denominator_columns

    @field_validator("new_column_names", mode="after")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            numerator_columns = info.data.get("numerator_columns", [])
            if len(new_column_names) != len(numerator_columns):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of numerator_columns ({len(numerator_columns)})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "ConcentrationIndexFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        ConcentrationIndexFeatures
            Fitted transformer instance.
        """
        default_names = [
            f"{num}__conc__{'__'.join(denoms)}"
            for num, denoms in zip(self.numerator_columns, self.denominator_columns, strict=False)
        ]

        if self.new_column_names is None:
            self.new_column_names = default_names

        self._column_mapping = {d: [n] for d, n in zip(default_names, self.new_column_names, strict=False)}

        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating concentration index features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with concentration index features appended.
        """
        new_columns = []

        for num_col, denom_cols in zip(self.numerator_columns, self.denominator_columns, strict=False):
            default_name = f"{num_col}__conc__{'__'.join(denom_cols)}"
            new_col_name = self._column_mapping[default_name][0]

            denom_sum = pl.sum_horizontal([pl.col(d) for d in denom_cols])
            if self.smoothing:
                denom_sum = denom_sum + 1

            new_columns.append((pl.col(num_col) / denom_sum).alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list(
                {
                    col
                    for num, denoms in zip(self.numerator_columns, self.denominator_columns, strict=False)
                    for col in [num] + denoms
                }
            )
            X = X.drop(columns_to_drop)

        return X
