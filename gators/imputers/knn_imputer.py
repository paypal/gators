from typing import Literal

import polars as pl
from pydantic import PositiveInt, PrivateAttr

from ..transformer._base_transformer import _BaseTransformer

_NUMERIC_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
)


class KNNImputer(_BaseTransformer):
    """Impute missing values using K-Nearest Neighbors.

    For each row with missing values, finds the ``n_neighbors`` closest rows in
    the training set (using Euclidean distance over the non-null numeric
    columns) and fills the missing values with the mean (``weights='uniform'``)
    or inverse-distance-weighted mean (``weights='distance'``) of those
    neighbors' values.

    The algorithm is implemented entirely in Polars:

    1. **Fit**: retain all training rows that are complete in both ``subset``
       and the feature columns; store them in ``_train_df_`` as the reference
       pool.  Compute per-column medians as a global fallback.
    2. **Transform**: rows with no nulls pass through unchanged.  Rows with
       nulls are grouped by their *missing pattern* (which columns are null)
       so that each group can share a single cross-join + distance computation
       with the reference pool.

    .. note::
        KNN distance is scale-sensitive.  Consider adding a
        :class:`~gators.scalers.StandardScaler` (or similar) before this
        imputer when columns have very different magnitudes.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use for imputation.
    subset : list[str] or None, default=None
        Numeric columns to impute.  If ``None``, all numeric columns that
        contain at least one null in the training data are selected
        automatically.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function applied when aggregating neighbor values.

        - ``'uniform'``: all neighbors contribute equally (plain mean).
        - ``'distance'``: neighbors are weighted by ``1 / (d + ε)`` so that
          closer neighbors have a stronger influence.

    Attributes
    ----------
    _train_df_ : pl.DataFrame
        Complete training rows used as the KNN reference pool.
    _global_stats_ : dict[str, float]
        Per-column median fallback values, used when no feature columns are
        available to compute distances.
    _feature_cols_ : list[str]
        Numeric columns (not in ``subset``) used to compute pairwise distances.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.imputers import KNNImputer

    >>> X = pl.DataFrame({
    ...     "age":    [25.0, 30.0, 35.0, 40.0, None],
    ...     "salary": [50.0, 60.0, 70.0, 80.0, 75.0],
    ... })
    >>> imputer = KNNImputer(n_neighbors=2, subset=["age"])
    >>> imputer.fit(X)
    KNNImputer(n_neighbors=2, subset=['age'], weights='uniform')
    >>> imputer.transform(X)
    shape: (5, 2)
    ┌──────┬────────┐
    │ age  ┆ salary │
    │ ---  ┆ ---    │
    │ f64  ┆ f64    │
    ╞══════╪════════╡
    │ 25.0 ┆ 50.0   │
    │ 30.0 ┆ 60.0   │
    │ 35.0 ┆ 70.0   │
    │ 40.0 ┆ 80.0   │
    │ 37.5 ┆ 75.0   │  ← mean of the 2 nearest neighbours (age=35, age=40)
    └──────┴────────┘
    """

    n_neighbors: PositiveInt = 5
    subset: list[str] | None = None
    weights: Literal["uniform", "distance"] = "uniform"

    _train_df_: pl.DataFrame = PrivateAttr(default_factory=pl.DataFrame)
    _global_stats_: dict[str, float] = PrivateAttr(default_factory=dict)
    _feature_cols_: list[str] = PrivateAttr(default_factory=list)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "KNNImputer":
        """Fit by storing the complete training rows as the KNN reference pool.

        Parameters
        ----------
        X : pl.DataFrame
            Training DataFrame with numeric columns.
        y : pl.Series or None, default=None
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        KNNImputer
            The fitted transformer instance.
        """
        dtype_map = dict(zip(X.columns, X.dtypes))

        if not self.subset:
            self.subset = [
                col
                for col in X.columns
                if dtype_map[col] in _NUMERIC_DTYPES and X[col].null_count() > 0
            ]

        self._feature_cols_ = [
            col for col in X.columns if col not in self.subset and dtype_map[col] in _NUMERIC_DTYPES
        ]

        # Reference pool: rows with no nulls in either subset or feature cols
        all_cols = self._feature_cols_ + self.subset
        complete_mask = pl.all_horizontal([pl.col(c).is_not_null() for c in all_cols])
        self._train_df_ = X.filter(complete_mask).select(all_cols)

        # Global median fallback (used when no feature cols are available)
        source = self._train_df_ if len(self._train_df_) > 0 else X
        stats = source.select([pl.col(c).median() for c in self.subset]).row(0)
        self._global_stats_ = {
            col: (val if val is not None else 0.0) for col, val in zip(self.subset, stats)
        }

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Impute missing values in ``subset`` columns using KNN.

        Parameters
        ----------
        X : pl.DataFrame
            DataFrame to transform. May contain nulls in ``subset`` columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with nulls in ``subset`` columns filled.
        """
        if not self.subset:
            return X

        has_null = pl.any_horizontal([pl.col(c).is_null() for c in self.subset])
        X_indexed = X.with_row_index("__row_idx__")
        rows_to_impute = X_indexed.filter(has_null)

        if len(rows_to_impute) == 0:
            return X

        # Group rows by their missing pattern so each group can share one
        # cross-join with the training pool.
        pattern_expr = pl.concat_str(
            [pl.col(c).is_null().cast(pl.String) for c in self.subset],
            separator=",",
        ).alias("__pattern__")
        rows_to_impute = rows_to_impute.with_columns(pattern_expr)

        processed_groups: list[pl.DataFrame] = []
        for pattern_val in rows_to_impute["__pattern__"].unique().to_list():
            group = rows_to_impute.filter(pl.col("__pattern__") == pattern_val).drop("__pattern__")

            fill_cols = [c for c in self.subset if group[c].null_count() > 0]
            # Only use feature columns that are non-null in the entire group
            avail_features = [c for c in self._feature_cols_ if group[c].null_count() == 0]

            if not avail_features or len(self._train_df_) == 0:
                group = group.with_columns(
                    [pl.col(c).fill_null(self._global_stats_[c]) for c in fill_cols]
                )
            else:
                group = self._impute_group(group, fill_cols, avail_features)

            processed_groups.append(group)

        imputed = pl.concat(processed_groups)
        not_imputed = X_indexed.filter(~has_null)

        return pl.concat([not_imputed, imputed]).sort("__row_idx__").drop("__row_idx__")

    def _impute_group(
        self,
        group: pl.DataFrame,
        fill_cols: list[str],
        feature_cols: list[str],
    ) -> pl.DataFrame:
        """KNN-impute ``fill_cols`` for a single missing-pattern group.

        All rows in ``group`` share the same set of null columns (``fill_cols``).
        Distance is computed over ``feature_cols`` using normalised Euclidean
        distance (divided by the number of features so distances are comparable
        across groups with different numbers of available features).
        """
        # Rename training columns to avoid name collisions in the cross-join.
        train_renamed = self._train_df_.select(feature_cols + fill_cols).rename(
            {c: f"__t_{c}__" for c in feature_cols + fill_cols}
        )

        cross = group.select(["__row_idx__"] + feature_cols).join(train_renamed, how="cross")

        n_feat = len(feature_cols)
        cross = cross.with_columns(
            (
                pl.sum_horizontal([(pl.col(c) - pl.col(f"__t_{c}__")).pow(2) for c in feature_cols])
                / n_feat
            )
            .sqrt()
            .alias("__dist__")
        )

        # Rank distances within each query row (rank=1 is the nearest neighbour).
        # Using "ordinal" so ties always produce distinct, consecutive ranks.
        cross = cross.with_columns(
            pl.col("__dist__").rank("ordinal").over("__row_idx__").alias("__rank__")
        )
        top_k = cross.filter(pl.col("__rank__") <= self.n_neighbors)

        if self.weights == "uniform":
            agg = top_k.group_by("__row_idx__").agg(
                [pl.col(f"__t_{c}__").mean().alias(c) for c in fill_cols]
            )
        else:
            # Inverse-distance weights: w = 1 / (d + ε)
            top_k = top_k.with_columns((1.0 / (pl.col("__dist__") + 1e-10)).alias("__w__"))
            w_sum = top_k.group_by("__row_idx__").agg(pl.col("__w__").sum().alias("__wsum__"))
            top_k = top_k.join(w_sum, on="__row_idx__")
            agg = top_k.group_by("__row_idx__").agg(
                [
                    (pl.col(f"__t_{c}__") * pl.col("__w__") / pl.col("__wsum__")).sum().alias(c)
                    for c in fill_cols
                ]
            )

        # Drop the null fill_cols from group, join imputed values back,
        # then restore the original column order.
        return group.drop(fill_cols).join(agg, on="__row_idx__").select(group.columns)
