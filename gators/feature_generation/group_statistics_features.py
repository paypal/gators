from typing import Any

import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer

AGGREGATION_FUNCTIONS = [
    "mean",
    "std",
    "median",
    "min",
    "max",
    "sum",
    "count",
    "count_null",
    "range",
    "mean_ratio",
    "median_ratio",
    "zscore",
    "minmax",
    "rank",
    "pct_rank",
]


class GroupStatisticsFeatures(_BaseTransformer):
    """
    Generates group-level feature columns for numerical columns.

    Two categories of output are supported:

    **Absolute statistics** — the raw group aggregate is appended as a new column:

    - ``'mean'``: group mean
    - ``'std'``: group standard deviation
    - ``'median'``: group median
    - ``'min'``: group minimum
    - ``'max'``: group maximum
    - ``'sum'``: group sum
    - ``'count'``: group count (nulls excluded)
    - ``'count_null'``: group count of null values
    - ``'range'``: group range (max − min)

    **Relative statistics** — each row is scaled by its group aggregate:

    - ``'mean_ratio'``: value / group_mean
    - ``'median_ratio'``: value / group_median
    - ``'zscore'``: (value − group_mean) / group_std
    - ``'minmax'``: (value − group_min) / (group_max − group_min)

    **Order statistics** — a row's position within its group. Unlike the two
    categories above, these are computed *live* on the batch passed to
    ``transform()`` (via Polars' ``.rank().over()``) rather than looked up from
    statistics frozen at ``fit()`` time, since a rank is inherently relative to
    every other row in the same call, not a fixed group aggregate:

    - ``'rank'``: average-tie rank of the value within its group (1-based)
    - ``'pct_rank'``: rank divided by the (non-null) group size, in ``(0, 1]``

    Both categories can be combined freely in a single transformer call.
    All generated columns follow the naming pattern
    ``'{func}_{num_col}__per_{groupby_col}'``.

    Importance for Fraud Detection
    --------------------------------
    Group features are particularly valuable in fraud detection because they capture
    how a transaction compares to the typical behaviour of its segment (merchant,
    customer, time-of-day, geography, etc.).

    - **Absolute stats** provide context: "the average transaction for merchant A is $183".
    - **Relative stats** expose anomalies: "this transaction is 10× the group average"
      (``mean_ratio``), "3σ above the group mean" (``zscore``), or "at the very top of
      the observed range" (``minmax``).

    Parameters
    ----------
    subset : list[str]
        List of numerical column names to aggregate.
    by : list[str]
        List of column names to use for groupby operations. Each column is used
        for a *separate* groupby (e.g., ``['cat1', 'cat2']`` creates features
        grouped by ``cat1`` and separate features grouped by ``cat2``).
    func : list[str]
        List of functions to apply. Any mix of absolute and relative functions
        listed above is valid.
    fill_value : float, default=0.0
        Value used when the denominator is zero or null for relative statistics
        (``'mean_ratio'``, ``'median_ratio'``, ``'zscore'``, ``'minmax'``).
        Has no effect on absolute statistics.
    drop_columns : bool, default=False
        Whether to drop the original numerical columns after creating features.
    new_column_names : list[str], default=None
        Custom names for the generated columns. If ``None``, names are
        auto-generated as ``'{func}_{num_col}__per_{groupby_col}'``. Must have
        the same length as ``subset × by × func``.

    Examples
    --------
    >>> from gators.feature_generation import GroupStatisticsFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'amount': [100, 200, 150, 300, 250],
    ...     'cat1': ['A', 'A', 'B', 'B', 'A'],
    ...     'cat2': ['X', 'Y', 'X', 'X', 'X'],
    ... })

    **Example 1: Absolute statistics**

    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['mean', 'count'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result
    shape: (5, 5)
    ┌────────┬──────┬──────┬───────────────────────┬────────────────────────┐
    │ amount ┆ cat1 ┆ cat2 ┆ mean_amount__per_cat1 ┆ count_amount__per_cat1 │
    │ ---    ┆ ---  ┆ ---  ┆ ---                   ┆ ---                    │
    │ i64    ┆ str  ┆ str  ┆ f64                   ┆ u32                    │
    ╞════════╪══════╪══════╪═══════════════════════╪════════════════════════╡
    │ 100    ┆ A    ┆ X    ┆ 183.333333            ┆ 3                      │
    │ 200    ┆ A    ┆ Y    ┆ 183.333333            ┆ 3                      │
    │ 150    ┆ B    ┆ X    ┆ 225.0                 ┆ 2                      │
    │ 300    ┆ B    ┆ X    ┆ 225.0                 ┆ 2                      │
    │ 250    ┆ A    ┆ X    ┆ 183.333333            ┆ 3                      │
    └────────┴──────┴──────┴───────────────────────┴────────────────────────┘

    **Example 2: Relative statistics**

    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['mean_ratio', 'zscore'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['amount', 'cat1', 'cat2', 'mean_ratio_amount__per_cat1', 'zscore_amount__per_cat1']

    **Example 3: Mixing absolute and relative**

    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1'],
    ...     func=['mean', 'zscore', 'minmax'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['amount', 'cat1', 'cat2', 'mean_amount__per_cat1', 'zscore_amount__per_cat1', 'minmax_amount__per_cat1']

    **Example 4: Multiple groupby columns**

    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['amount'],
    ...     by=['cat1', 'cat2'],
    ...     func=['mean'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['amount', 'cat1', 'cat2', 'mean_amount__per_cat1', 'mean_amount__per_cat2']

    **Example 5: Zero denominator handling (relative statistics)**

    >>> X_zero = pl.DataFrame({'v': [0, 0, 5, 10], 'g': ['A', 'A', 'B', 'B']})
    >>> transformer = GroupStatisticsFeatures(
    ...     subset=['v'],
    ...     by=['g'],
    ...     func=['mean_ratio'],
    ...     fill_value=-1.0,
    ... )
    >>> result = transformer.fit_transform(X_zero)
    >>> result['mean_ratio_v__per_g'][0]  # group A mean=0 -> fill_value
    -1.0
    """

    subset: list[str]
    by: list[str]
    func: list[str]
    fill_value: float = 0.0
    drop_columns: bool = False
    new_column_names: list[str] | None = None
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    # Keyed (num_col, groupby_col) → {stat_name: {group_value: float}}
    _group_stats: dict = PrivateAttr(default_factory=dict)

    @field_validator("func")
    def check_func(cls, func):
        for fun in func:
            if fun not in AGGREGATION_FUNCTIONS:
                raise ValueError(
                    f"{fun} is not in the predefined list of aggregation functions: {AGGREGATION_FUNCTIONS}"
                )
        return func

    @field_validator("new_column_names")
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            subset = info.data.get("subset", [])
            by = info.data.get("by", [])
            func = info.data.get("func", [])
            expected_length = len(subset) * len(by) * len(func)
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match the total number of features created ({expected_length}). "
                    f"Expected: {len(subset)} numerical columns × {len(by)} groupby columns × {len(func)} func = {expected_length}"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "GroupStatisticsFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        GroupStatisticsFeatures
            Fitted transformer instance.
        """
        default_names = []
        for num_col in self.subset:
            for groupby_col in self.by:
                for fun in self.func:
                    default_names.append(f"{fun}_{num_col}__per_{groupby_col}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = {d: [n] for d, n in zip(default_names, self.new_column_names, strict=False)}

        # Compute and store group stats needed by all requested functions.
        _NEEDED: dict[str, list[str]] = {
            "mean": ["mean"], "std": ["std"], "median": ["median"],
            "min": ["min"], "max": ["max"], "sum": ["sum"], "count": ["count"],
            "count_null": ["count_null"],
            "range": ["min", "max"],
            "mean_ratio": ["mean"], "median_ratio": ["median"],
            "zscore": ["mean", "std"], "minmax": ["min", "max"],
            "rank": [], "pct_rank": [],
        }
        _POLARS_AGG = {
            "mean": lambda c: pl.col(c).mean(),
            "std":  lambda c: pl.col(c).std(),
            "median": lambda c: pl.col(c).median(),
            "min":  lambda c: pl.col(c).min(),
            "max":  lambda c: pl.col(c).max(),
            "sum":  lambda c: pl.col(c).sum(),
            "count": lambda c: pl.col(c).count().cast(pl.Float64),
            "count_null": lambda c: pl.col(c).is_null().sum().cast(pl.Float64),
        }
        for num_col in self.subset:
            for groupby_col in self.by:
                needed = {s for f in self.func for s in _NEEDED[f]}
                aggs = [_POLARS_AGG[s](num_col).alias(s) for s in needed]
                gdf = X.group_by(groupby_col).agg(aggs)
                self._group_stats[(num_col, groupby_col)] = {
                    s: {
                        str(row[groupby_col]): float(row[s]) if row[s] is not None else float("nan")
                        for row in gdf.iter_rows(named=True)
                    }
                    for s in needed
                }

        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating group statistic features.

        Applies the group statistics learned during ``fit()`` (via a left join on the
        groupby column) rather than recomputing statistics live from ``X`` - this keeps
        train/test behavior consistent (mirroring the ONNX export's frozen lookup) and
        correctly falls back to ``fill_value``/null for groups unseen during ``fit()``.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with group statistic features appended.
        """
        _NEEDED: dict[str, list[str]] = {
            "mean": ["mean"], "std": ["std"], "median": ["median"],
            "min": ["min"], "max": ["max"], "sum": ["sum"], "count": ["count"],
            "count_null": ["count_null"],
            "range": ["min", "max"],
            "mean_ratio": ["mean"], "median_ratio": ["median"],
            "zscore": ["mean", "std"], "minmax": ["min", "max"],
            "rank": [], "pct_rank": [],
        }

        # Join the fitted group statistics onto X (one join per num_col x groupby_col pair).
        # 'rank'/'pct_rank' need no join (they're computed live below), so pairs whose
        # func list only contains those are skipped here entirely.
        join_columns: list[str] = []
        stat_col_name: dict[tuple[str, str, str], str] = {}
        for num_col in self.subset:
            for groupby_col in self.by:
                needed = sorted({s for f in self.func for s in _NEEDED[f]})
                if not needed:
                    continue
                stats = self._group_stats.get((num_col, groupby_col), {})
                key_col = f"__gsf_key__{num_col}__{groupby_col}"
                keys = list(next(iter(stats.values()), {}).keys())
                lookup_data: dict[str, Any] = {key_col: pl.Series(key_col, keys, dtype=pl.String)}
                for s in needed:
                    helper_col = f"__gsf__{num_col}__{groupby_col}__{s}"
                    stat_col_name[(num_col, groupby_col, s)] = helper_col
                    mapping = stats.get(s, {})
                    lookup_data[helper_col] = pl.Series(
                        helper_col, [mapping.get(k, float("nan")) for k in keys], dtype=pl.Float64
                    )
                    join_columns.append(helper_col)
                lookup = pl.DataFrame(lookup_data)
                X = X.join(
                    lookup,
                    left_on=pl.col(groupby_col).cast(pl.String),
                    right_on=key_col,
                    how="left",
                )

        new_columns = []

        for num_col in self.subset:
            for groupby_col in self.by:
                for fun in self.func:
                    default_name = f"{fun}_{num_col}__per_{groupby_col}"
                    new_col_name = self._column_mapping[default_name][0]

                    if fun == "mean_ratio":
                        denominator = pl.col(stat_col_name[(num_col, groupby_col, "mean")])
                        expr = (
                            pl.when((denominator == 0) | denominator.is_null())
                            .then(self.fill_value)
                            .otherwise(pl.col(num_col).cast(pl.Float64) / denominator)
                            .alias(new_col_name)
                        )
                    elif fun == "median_ratio":
                        denominator = pl.col(stat_col_name[(num_col, groupby_col, "median")])
                        expr = (
                            pl.when((denominator == 0) | denominator.is_null())
                            .then(self.fill_value)
                            .otherwise(pl.col(num_col).cast(pl.Float64) / denominator)
                            .alias(new_col_name)
                        )
                    elif fun == "zscore":
                        group_mean = pl.col(stat_col_name[(num_col, groupby_col, "mean")])
                        group_std = pl.col(stat_col_name[(num_col, groupby_col, "std")])
                        expr = (
                            pl.when((group_std == 0) | group_std.is_null())
                            .then(self.fill_value)
                            .otherwise((pl.col(num_col).cast(pl.Float64) - group_mean) / group_std)
                            .alias(new_col_name)
                        )
                    elif fun == "minmax":
                        group_min = pl.col(stat_col_name[(num_col, groupby_col, "min")])
                        group_max = pl.col(stat_col_name[(num_col, groupby_col, "max")])
                        range_val = group_max - group_min
                        expr = (
                            pl.when((range_val == 0) | range_val.is_null())
                            .then(self.fill_value)
                            .otherwise((pl.col(num_col).cast(pl.Float64) - group_min) / range_val)
                            .alias(new_col_name)
                        )
                    elif fun == "rank":
                        # Order statistic: computed live on the current batch, not looked up
                        # from fit()-time stats (a rank only makes sense relative to the
                        # other rows actually present in this call).
                        expr = (
                            pl.col(num_col)
                            .cast(pl.Float64)
                            .rank(method="average")
                            .over(groupby_col)
                            .alias(new_col_name)
                        )
                    elif fun == "pct_rank":
                        rank_expr = (
                            pl.col(num_col).cast(pl.Float64).rank(method="average").over(groupby_col)
                        )
                        group_size = pl.col(num_col).count().over(groupby_col).cast(pl.Float64)
                        expr = (rank_expr / group_size).alias(new_col_name)
                    else:
                        if fun == "range":
                            expr = (
                                pl.col(stat_col_name[(num_col, groupby_col, "max")])
                                - pl.col(stat_col_name[(num_col, groupby_col, "min")])
                            ).alias(new_col_name)
                        else:
                            expr = pl.col(stat_col_name[(num_col, groupby_col, fun)]).alias(new_col_name)

                    new_columns.append(expr)

        X = X.with_columns(new_columns)
        X = X.drop(join_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
