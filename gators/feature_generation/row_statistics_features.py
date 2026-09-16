import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

AGGREGATION_FUNCTIONS = ["min", "max", "mean", "median", "std", "range", "sum", "count"]


class RowStatisticsFeatures(_BaseTransformer):
    """
    Generates row-level aggregation features across groups of columns.

    This transformer computes statistics (min, max, mean, median, std, range, sum, count)
    horizontally across specified column groups for each row. Unlike
    GroupStatisticsFeatures which aggregates vertically (across rows within groups),
    this computes statistics across columns within each row.

    Auto-generated column names follow the pattern ``'{group_name}__{func}'``.

    Parameters
    ----------
    column_groups : dict[str, list[str]]
        Dictionary mapping group names to lists of column names. Each group defines
        a set of columns over which to compute row-level statistics. Every list must
        contain at least 2 columns.
        Example: ``{'card_fields': ['card1', 'card2', 'card3']}``
    func : list[str]
        Aggregation functions to apply to every group. Available options:

        - ``'min'``: Row-wise minimum
        - ``'max'``: Row-wise maximum
        - ``'mean'``: Row-wise mean
        - ``'median'``: Row-wise median
        - ``'std'``: Row-wise standard deviation
        - ``'range'``: Row-wise range (max − min)
        - ``'sum'``: Row-wise sum
        - ``'count'``: Row-wise count of non-null values
    drop_columns : bool, default=False
        Whether to drop the original columns after creating aggregation features.
    new_column_names : list[str], default=None
        Custom names for the generated columns. If ``None``, names are
        auto-generated as ``'{group_name}__{func}'``. Must have the same length as
        ``len(column_groups) × len(func)``.

    Examples
    --------
    >>> from gators.feature_generation import RowStatisticsFeatures
    >>> import polars as pl

    **Example 1: Single group with multiple aggregations**

    >>> X = pl.DataFrame({'A': [9, 9, 7], 'B': [3, 4, 5], 'C': [6, 7, 8]})
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'cluster_1': ['A', 'B']},
    ...     func=['mean', 'std'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.select(['cluster_1__mean', 'cluster_1__std'])
    shape: (3, 2)
    ┌─────────────────┬────────────────┐
    │ cluster_1__mean ┆ cluster_1__std │
    │ ---             ┆ ---            │
    │ f64             ┆ f64            │
    ╞═════════════════╪════════════════╡
    │ 6.0             ┆ 4.242641       │
    │ 6.5             ┆ 3.535534       │
    │ 6.0             ┆ 1.414214       │
    └─────────────────┴────────────────┘

    **Example 2: Multiple groups**

    >>> X = pl.DataFrame({'A': [9, 9, 7], 'B': [3, 4, 5], 'C': [6, 7, 8], 'D': [1, 2, 3]})
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'cluster_1': ['A', 'B'], 'cluster_2': ['C', 'D']},
    ...     func=['min', 'max'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['A', 'B', 'C', 'D', 'cluster_1__min', 'cluster_1__max', 'cluster_2__min', 'cluster_2__max']

    **Example 3: Custom column names**

    >>> X = pl.DataFrame({'amount1': [100, 200, 150], 'amount2': [50, 100, 75], 'amount3': [25, 50, 30]})
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'amounts': ['amount1', 'amount2', 'amount3']},
    ...     func=['mean', 'std'],
    ...     new_column_names=['avg_amount', 'std_amount'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'avg_amount' in result.columns
    True

    **Example 4: Fraud detection — card verification fields**

    >>> X = pl.DataFrame({
    ...     'card_cvv_match':  [1, 0, 1, 1],
    ...     'card_addr_match': [1, 1, 0, 1],
    ...     'card_zip_match':  [1, 1, 1, 0],
    ...     'is_fraud':        [0, 1, 1, 1],
    ... })
    >>> transformer = RowStatisticsFeatures(
    ...     column_groups={'verification': ['card_cvv_match', 'card_addr_match', 'card_zip_match']},
    ...     func=['mean', 'std'],
    ...     new_column_names=['verif__mean', 'verif__std'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result['verif__mean'][0]   # legitimate: all checks pass
    1.0
    >>> result['verif__std'][0]    # legitimate: no variance
    0.0
    """

    column_groups: dict[str, list[str]]
    func: list[str]
    drop_columns: bool = False
    new_column_names: list[str] | None = None
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @field_validator("column_groups")
    def check_column_groups(cls, column_groups):
        if not column_groups:
            raise ValueError("column_groups cannot be empty")
        for key, val in column_groups.items():
            if not isinstance(val, list):
                raise TypeError(f"column_groups['{key}'] must be a list")
            if len(val) < 2:
                raise ValueError(
                    f"column_groups['{key}'] must contain at least 2 columns for row-level aggregation"
                )
        return column_groups

    @field_validator("func")
    def check_func(cls, func):
        for f in func:
            if f not in AGGREGATION_FUNCTIONS:
                raise ValueError(
                    f"{f} is not in the predefined list of aggregation functions: {AGGREGATION_FUNCTIONS}"
                )
        return func

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            column_groups = info.data.get("column_groups", {})
            func = info.data.get("func", [])
            expected_length = len(column_groups) * len(func)
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match the total number of features created ({expected_length})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "RowStatisticsFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        RowStatisticsFeatures
            Fitted transformer instance.
        """
        default_names = [
            f"{group_name}__{f}" for group_name in self.column_groups for f in self.func
        ]

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = {d: [n] for d, n in zip(default_names, self.new_column_names, strict=False)}

        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating row-level aggregation features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with row-level aggregation features appended.
        """
        new_columns = []
        columns_to_drop = set()

        for group_name, cols in self.column_groups.items():
            for f in self.func:
                default_name = f"{group_name}__{f}"
                new_col_name = self._column_mapping[default_name][0]

                if f == "mean":
                    expr = pl.concat_list(cols).list.mean().alias(new_col_name)
                elif f == "std":
                    expr = pl.concat_list(cols).list.std().alias(new_col_name)
                elif f == "median":
                    expr = pl.concat_list(cols).list.median().alias(new_col_name)
                elif f == "min":
                    expr = pl.concat_list(cols).list.min().alias(new_col_name)
                elif f == "max":
                    expr = pl.concat_list(cols).list.max().alias(new_col_name)
                elif f == "range":
                    expr = (
                        pl.concat_list(cols).list.max() - pl.concat_list(cols).list.min()
                    ).alias(new_col_name)
                elif f == "sum":
                    expr = pl.concat_list(cols).list.sum().alias(new_col_name)
                elif f == "count":
                    expr = (
                        pl.concat_list(cols).list.drop_nulls().list.len().cast(pl.Float64)
                    ).alias(new_col_name)

                new_columns.append(expr)

            if self.drop_columns:
                columns_to_drop.update(cols)

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(list(columns_to_drop))

        return X
