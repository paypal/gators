import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer


class AsymmetryIndexFeatures(_BaseTransformer):
    """
    Generates asymmetry index features for pairs of columns.

    For each pair (x_i, y_i), computes:

        ``asym(x, y) = (x / y) / (y / x + 1)  =  x² / (y · (x + y))``

    Properties:

    * ``asym = 0.5`` when ``x == y`` (perfectly symmetric).
    * ``asym > 0.5`` when ``x > y`` (x dominates); can be greater than 1.
    * ``asym < 0.5`` when ``x < y`` (y dominates).

    With ``smoothing=True`` (default), ``1`` is added to both ``x`` and ``y``
    before computing the ratio, preventing division by zero.

    Parameters
    ----------
    x_columns : list[str]
        List of column names to use as the numerator side. One asymmetry index
        is created per entry.
    y_columns : list[str]
        List of column names to use as the denominator side. Must have the same
        length as ``x_columns``.
    smoothing : bool, optional
        Whether to add ``1`` to both ``x`` and ``y`` before computing the index,
        by default ``True``.
    new_column_names : list[str], optional
        Custom output column names. If ``None``, names are auto-generated as
        ``'{x}__asym__{y}'``, by default ``None``.
    drop_columns : bool, optional
        Whether to drop the original ``x`` and ``y`` columns after creating the
        asymmetry features, by default ``False``.

    Examples
    --------
    >>> from gators.feature_generation import AsymmetryIndexFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'clicks_a': [10, 0, 50],
    ...     'clicks_b': [10, 20, 5],
    ... })

    **Example 1: Basic usage (with smoothing)**

    >>> transformer = AsymmetryIndexFeatures(
    ...     x_columns=['clicks_a'],
    ...     y_columns=['clicks_b'],
    ... )
    >>> transformer.fit(X)
    AsymmetryIndexFeatures(x_columns=['clicks_a'], y_columns=['clicks_b'], smoothing=True)
    >>> result = transformer.transform(X)
    >>> result
    shape: (3, 3)
    ┌──────────┬──────────┬──────────────────────────┐
    │ clicks_a │ clicks_b │ clicks_a__asym__clicks_b │
    │ i64      │ i64      │ f64                      │
    ╞══════════╪══════════╪══════════════════════════╡
    │ 10       │ 10       │ 0.5                      │
    │ 0        │ 20       │ 0.002165                 │
    │ 50       │ 5        │ 7.605263                 │
    └──────────┴──────────┴──────────────────────────┘

    **Example 2: No smoothing**

    When ``smoothing=False``, equal-zero rows produce ``NaN``.

    >>> transformer = AsymmetryIndexFeatures(
    ...     x_columns=['clicks_a'],
    ...     y_columns=['clicks_b'],
    ...     smoothing=False,
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: Multiple pairs**

    >>> X2 = pl.DataFrame({
    ...     'views_a': [100, 200, 0],
    ...     'views_b': [50,  200, 0],
    ...     'sales_a': [10,  30,  5],
    ...     'sales_b': [40,  30, 15],
    ... })
    >>> transformer = AsymmetryIndexFeatures(
    ...     x_columns=['views_a', 'sales_a'],
    ...     y_columns=['views_b', 'sales_b'],
    ... )
    >>> result = transformer.fit_transform(X2)

    **Example 4: Custom column names**

    >>> transformer = AsymmetryIndexFeatures(
    ...     x_columns=['clicks_a'],
    ...     y_columns=['clicks_b'],
    ...     new_column_names=['click_asym'],
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 5: drop_columns=True**

    >>> transformer = AsymmetryIndexFeatures(
    ...     x_columns=['clicks_a'],
    ...     y_columns=['clicks_b'],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['clicks_a__asym__clicks_b']
    """

    x_columns: list[str]
    y_columns: list[str]
    smoothing: bool = True
    new_column_names: list[str] | None = None
    drop_columns: bool = False
    _column_mapping: dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("y_columns", mode="after")
    @classmethod
    def check_lengths_match(cls, y_columns, info):
        x_columns = info.data.get("x_columns", [])
        if len(x_columns) != len(y_columns):
            raise ValueError(
                f"Length of x_columns ({len(x_columns)}) "
                f"must match length of y_columns ({len(y_columns)})"
            )
        return y_columns

    @field_validator("new_column_names", mode="after")
    @classmethod
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            x_columns = info.data.get("x_columns", [])
            if len(new_column_names) != len(x_columns):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of x_columns ({len(x_columns)})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "AsymmetryIndexFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        AsymmetryIndexFeatures
            Fitted transformer instance.
        """
        default_names = [
            f"{x_col}__asym__{y_col}" for x_col, y_col in zip(self.x_columns, self.y_columns)
        ]

        if self.new_column_names is None:
            self.new_column_names = default_names

        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating asymmetry index features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with asymmetry index features appended.
        """
        if not self._column_mapping:
            return X

        new_columns = []

        for x_col, y_col in zip(self.x_columns, self.y_columns):
            default_name = f"{x_col}__asym__{y_col}"
            new_col_name = self._column_mapping[default_name]

            if self.smoothing:
                x_expr = pl.col(x_col).cast(pl.Float64) + 1
                y_expr = pl.col(y_col).cast(pl.Float64) + 1
            else:
                x_expr = pl.col(x_col).cast(pl.Float64)
                y_expr = pl.col(y_col).cast(pl.Float64)

            # asym(x, y) = x² / (y * (x + y))
            asym_expr = (x_expr**2) / (y_expr * (x_expr + y_expr))
            new_columns.append(asym_expr.alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list(
                {
                    col
                    for x_col, y_col in zip(self.x_columns, self.y_columns)
                    for col in [x_col, y_col]
                }
            )
            X = X.drop(columns_to_drop)

        return X
