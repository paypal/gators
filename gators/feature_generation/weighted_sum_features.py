import polars as pl
from pydantic import PrivateAttr, field_validator

from ..transformer._base_transformer import _BaseTransformer


class WeightedSumFeatures(_BaseTransformer):
    """
    Generates weighted sum features from groups of columns.

    Each feature is computed as:

    .. math::

        f = b_0 + \\sum_i a_i \\cdot x_i

    where :math:`a_i` are per-column scalar weights (defaulting to ``1.0``) and
    :math:`b_0` is an optional bias term (defaulting to ``0.0``).

    This is the numerator component of :class:`GeneralizedRatioFeatures` exposed
    as a standalone transformer.  It is useful for applying pre-computed regression
    coefficients, domain-defined scoring weights, or any fixed linear projection
    to a group of columns without dividing by a denominator.

    Parameters
    ----------
    column_groups : list[list[str]]
        One inner list of column names per output feature.  The weighted sum
        of each inner group becomes the corresponding output column.
        Each inner list must be non-empty.
    coefficients : list[list[float]], optional
        Scalar weights applied to each column within its group.  Must mirror
        the shape of ``column_groups``.  Defaults to ``1.0`` for every column
        when ``None``.
    biases : list[float], optional
        Scalar intercept added to each weighted sum after all column terms are
        combined.  Must have the same length as ``column_groups``.  Defaults
        to ``0.0`` for every feature when ``None``.
    new_column_names : list[str], optional
        Custom output column names.  If ``None``, names are auto-generated as
        ``'{col1}__{col2}__...____wsum'``.
    drop_columns : bool, optional
        Whether to drop all source columns after creating the features,
        by default ``False``.

    Examples
    --------
    >>> from gators.feature_generation import WeightedSumFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'f1':  [5.0,  20.0,  0.0],
    ...     'f3':  [500.0, 200.0, 0.0],
    ...     'f2':  [60.0, 100.0, 10.0],
    ...     'amt_1h':  [6000.0, 4000.0, 0.0],
    ... })

    **Example 1: Simple sum of a group (equal weights)**

    >>> transformer = WeightedSumFeatures(
    ...     column_groups=[['f1', 'f3']],
    ... )
    >>> transformer.fit(X)
    WeightedSumFeatures(column_groups=[['f1', 'f3']])
    >>> result = transformer.transform(X)
    >>> 'f1__f3__wsum' in result.columns
    True

    **Example 2: Custom coefficients**

    Apply regression-style weights: weight amount twice as much as count.

    >>> transformer = WeightedSumFeatures(
    ...     column_groups=[['f1', 'f3']],
    ...     coefficients=[[1.0, 2.0]],
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: With bias term**

    >>> transformer = WeightedSumFeatures(
    ...     column_groups=[['f1', 'f3']],
    ...     coefficients=[[0.5, 0.1]],
    ...     biases=[-10.0],
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 4: Multiple output features**

    >>> transformer = WeightedSumFeatures(
    ...     column_groups=[['f1', 'f2'], ['f3', 'amt_1h']],
    ...     coefficients=[[1.0, -1.0], [1.0, -1.0]],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'f1__f2__wsum' in result.columns
    True
    >>> 'f3__amt_1h__wsum' in result.columns
    True

    **Example 5: Custom column names**

    >>> transformer = WeightedSumFeatures(
    ...     column_groups=[['f1', 'f3']],
    ...     new_column_names=['combined_score'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'combined_score' in result.columns
    True

    **Example 6: With drop_columns=True**

    >>> transformer = WeightedSumFeatures(
    ...     column_groups=[['f1', 'f2']],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'f1' in result.columns
    False
    >>> 'f2' in result.columns
    False
    """

    column_groups: list[list[str]]
    coefficients: list[list[float]] | None = None
    biases: list[float] | None = None
    new_column_names: list[str] | None = None
    drop_columns: bool = False
    _column_mapping: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    @field_validator("column_groups", mode="after")
    @classmethod
    def check_groups_non_empty(cls, column_groups):
        for i, group in enumerate(column_groups):
            if len(group) == 0:
                raise ValueError(f"column_groups[{i}] must contain at least one column name.")
        return column_groups

    @field_validator("coefficients", mode="after")
    @classmethod
    def check_coefficients_shape(cls, coefficients, info):
        if coefficients is not None:
            column_groups = info.data.get("column_groups", [])
            if len(coefficients) != len(column_groups):
                raise ValueError(
                    f"Length of coefficients ({len(coefficients)}) "
                    f"must match length of column_groups ({len(column_groups)})"
                )
            for i, (coeffs, cols) in enumerate(zip(coefficients, column_groups, strict=False)):
                if len(coeffs) != len(cols):
                    raise ValueError(
                        f"coefficients[{i}] has {len(coeffs)} entries "
                        f"but column_groups[{i}] has {len(cols)} columns."
                    )
        return coefficients

    @field_validator("biases", mode="after")
    @classmethod
    def check_biases_length(cls, biases, info):
        if biases is not None:
            column_groups = info.data.get("column_groups", [])
            if len(biases) != len(column_groups):
                raise ValueError(
                    f"Length of biases ({len(biases)}) "
                    f"must match length of column_groups ({len(column_groups)})"
                )
        return biases

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
        return "__".join(cols) + "__wsum"

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "WeightedSumFeatures":
        """Fit the transformer by resolving coefficients, biases, and column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        WeightedSumFeatures
            Fitted transformer instance.
        """
        if self.coefficients is None:
            self.coefficients = [[1.0] * len(cols) for cols in self.column_groups]

        if self.biases is None:
            self.biases = [0.0] * len(self.column_groups)

        default_names = [self._default_name(cols) for cols in self.column_groups]

        if self.new_column_names is None:
            self.new_column_names = default_names

        self._column_mapping = {d: [n] for d, n in zip(default_names, self.new_column_names, strict=False)}

        self._output_dtypes = {
            new: pl.Float64 for names in self._column_mapping.values() for new in names
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating weighted sum features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with weighted sum features appended.
        """
        new_columns = []

        assert self.coefficients is not None and self.biases is not None
        for cols, coeffs, bias in zip(self.column_groups, self.coefficients, self.biases, strict=False):
            default_name = self._default_name(cols)
            new_col_name = self._column_mapping[default_name][0]

            terms = [pl.col(col).cast(pl.Float64) * c for col, c in zip(cols, coeffs, strict=False)]
            expr = pl.sum_horizontal(terms) + bias

            new_columns.append(expr.alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list({col for group in self.column_groups for col in group})
            X = X.drop(columns_to_drop)

        return X
