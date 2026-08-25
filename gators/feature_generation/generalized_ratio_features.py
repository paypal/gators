import polars as pl
from pydantic import PositiveFloat, field_validator

from ..transformer._base_transformer import _BaseTransformer


class GeneralizedRatioFeatures(_BaseTransformer):
    """
    Generates ratio features from weighted sums of column groups.

    Each feature is computed as:

    .. math::

        r = \\frac{b_0 + \\sum_i a_i \\cdot x_i}{\\sum_j b_j \\cdot y_j + \\varepsilon}

    where :math:`a_i` and :math:`b_j` are per-column scalar weights (defaulting
    to ``1.0``), :math:`b_0` is an optional bias added to the numerator
    (defaulting to ``0.0``), and :math:`\\varepsilon` is a small constant added
    to the denominator to prevent division by zero.

    When ``numerator_columns[i]`` is an empty list (``[]``) and
    ``numerator_biases[i]`` is set, the numerator reduces to the constant
    :math:`b_0` — enabling expressions such as :math:`1/(f_1 + f_2 + \\varepsilon)`.

    Unlike :class:`ConcentrationIndexFeatures` (single numerator column, equal
    weights) or :class:`BurstFeatures` (one-to-one columns), this transformer
    allows **different numbers** of columns in the numerator and denominator
    groups and **independent coefficients** for each column in both groups.

    Parameters
    ----------
    numerator_columns : list[list[str]]
        One inner list of column names per output feature.  The weighted sum
        of each inner group forms the numerator of the corresponding ratio.
        An empty inner list (``[]``) is allowed when ``numerator_biases`` is
        provided, making the numerator a pure constant.
    denominator_columns : list[list[str]]
        One inner list of column names per output feature.  The weighted sum
        of each inner group forms the denominator.  Must have the same length
        as ``numerator_columns``.  Each inner list must be non-empty.
    numerator_coefficients : list[list[float]], optional
        Scalar weights applied to each numerator column.  Must mirror the
        shape of ``numerator_columns``.  Defaults to ``1.0`` for every column
        when ``None``.
    denominator_coefficients : list[list[float]], optional
        Scalar weights applied to each denominator column.  Must mirror the
        shape of ``denominator_columns``.  Defaults to ``1.0`` for every
        column when ``None``.
    numerator_biases : list[float], optional
        Scalar constant :math:`b_0` added to each numerator sum.  Must have
        the same length as ``numerator_columns``.  Defaults to ``0.0`` for
        every feature when ``None``.  Set to a non-zero value (and pass ``[]``
        as the corresponding inner list in ``numerator_columns``) to create a
        pure-constant numerator.
    epsilon : float, optional
        Small positive constant added to every denominator sum for numerical
        stability, by default ``1.0``.
    new_column_names : list[str], optional
        Custom output column names.  If ``None``, names are auto-generated as
        ``'{num1}__{num2}__...____gratio__{denom1}__{denom2}__...'``.
    drop_columns : bool, optional
        Whether to drop all source columns after creating the ratio features,
        by default ``False``.

    Examples
    --------
    >>> from gators.feature_generation import GeneralizedRatioFeatures
    >>> import polars as pl

    >>> X = pl.DataFrame({
    ...     'f1':  [5.0,  20.0,  0.0],
    ...     'f2':  [500.0, 200.0, 0.0],
    ...     'f3':  [60.0, 100.0, 10.0],
    ...     'f4':  [6000.0, 4000.0, 0.0],
    ... })

    **Example 1: Single ratio, equal weights**

    >>> transformer = GeneralizedRatioFeatures(
    ...     numerator_columns=[['f1', 'f2']],
    ...     denominator_columns=[['f3', 'f4']],
    ... )
    >>> transformer.fit(X)
    GeneralizedRatioFeatures(numerator_columns=[['f1', 'f2']], denominator_columns=[['f3', 'f4']])
    >>> result = transformer.transform(X)
    >>> 'f1__f2__gratio__f3__f4' in result.columns
    True

    **Example 2: Custom coefficients (different group sizes)**

    Weight the amount column twice as much as the count column in the numerator,
    and use a single-column denominator with a rescaling coefficient.

    >>> transformer = GeneralizedRatioFeatures(
    ...     numerator_columns=[['f1', 'f2']],
    ...     denominator_columns=[['f3']],
    ...     numerator_coefficients=[[1.0, 2.0]],
    ...     denominator_coefficients=[[0.5]],
    ... )
    >>> result = transformer.fit_transform(X)

    **Example 3: Multiple ratios**

    >>> transformer = GeneralizedRatioFeatures(
    ...     numerator_columns=[['f1'], ['f2']],
    ...     denominator_columns=[['f3'], ['f4']],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'f1__gratio__f3' in result.columns
    True
    >>> 'f2__gratio__f4' in result.columns
    True

    **Example 4: Custom column names**

    >>> transformer = GeneralizedRatioFeatures(
    ...     numerator_columns=[['f1', 'f2']],
    ...     denominator_columns=[['f3', 'f4']],
    ...     new_column_names=['composite_ratio'],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'composite_ratio' in result.columns
    True

    **Example 5: With drop_columns=True**

    >>> transformer = GeneralizedRatioFeatures(
    ...     numerator_columns=[['f1']],
    ...     denominator_columns=[['f3']],
    ...     drop_columns=True,
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'f1' in result.columns
    False
    >>> 'f3' in result.columns
    False

    **Example 6: Constant numerator — 1 / (f3 + f4 + epsilon)**

    Pass an empty inner list for ``numerator_columns`` and set
    ``numerator_biases`` to the desired constant.

    >>> transformer = GeneralizedRatioFeatures(
    ...     numerator_columns=[[]],
    ...     denominator_columns=[['f3', 'f4']],
    ...     numerator_biases=[1.0],
    ... )
    >>> result = transformer.fit_transform(X)
    >>> 'const__gratio__f3__f4' in result.columns
    True
    """

    numerator_columns: list[list[str]]
    denominator_columns: list[list[str]]
    numerator_coefficients: list[list[float]] | None = None
    denominator_coefficients: list[list[float]] | None = None
    numerator_biases: list[float] | None = None
    epsilon: PositiveFloat = 1.0
    new_column_names: list[str] | None = None
    drop_columns: bool = False

    @field_validator("denominator_columns", mode="after")
    @classmethod
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

    @field_validator("numerator_coefficients", mode="after")
    @classmethod
    def check_numerator_coefficients_shape(cls, numerator_coefficients, info):
        if numerator_coefficients is not None:
            numerator_columns = info.data.get("numerator_columns", [])
            if len(numerator_coefficients) != len(numerator_columns):
                raise ValueError(
                    f"Length of numerator_coefficients ({len(numerator_coefficients)}) "
                    f"must match length of numerator_columns ({len(numerator_columns)})"
                )
            for i, (coeffs, cols) in enumerate(zip(numerator_coefficients, numerator_columns, strict=False)):
                if len(coeffs) != len(cols):
                    raise ValueError(
                        f"numerator_coefficients[{i}] has {len(coeffs)} entries "
                        f"but numerator_columns[{i}] has {len(cols)} columns."
                    )
        return numerator_coefficients

    @field_validator("denominator_coefficients", mode="after")
    @classmethod
    def check_denominator_coefficients_shape(cls, denominator_coefficients, info):
        if denominator_coefficients is not None:
            denominator_columns = info.data.get("denominator_columns", [])
            if len(denominator_coefficients) != len(denominator_columns):
                raise ValueError(
                    f"Length of denominator_coefficients ({len(denominator_coefficients)}) "
                    f"must match length of denominator_columns ({len(denominator_columns)})"
                )
            for i, (coeffs, cols) in enumerate(zip(denominator_coefficients, denominator_columns, strict=False)):
                if len(coeffs) != len(cols):
                    raise ValueError(
                        f"denominator_coefficients[{i}] has {len(coeffs)} entries "
                        f"but denominator_columns[{i}] has {len(cols)} columns."
                    )
        return denominator_coefficients

    @field_validator("numerator_biases", mode="after")
    @classmethod
    def check_numerator_biases_length(cls, numerator_biases, info):
        if numerator_biases is not None:
            numerator_columns = info.data.get("numerator_columns", [])
            if len(numerator_biases) != len(numerator_columns):
                raise ValueError(
                    f"Length of numerator_biases ({len(numerator_biases)}) "
                    f"must match length of numerator_columns ({len(numerator_columns)})"
                )
        return numerator_biases

    @field_validator("new_column_names", mode="after")
    @classmethod
    def check_new_column_names_length(cls, new_column_names, info):
        if new_column_names is not None:
            numerator_columns = info.data.get("numerator_columns", [])
            if len(new_column_names) != len(numerator_columns):
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match length of numerator_columns ({len(numerator_columns)})"
                )
            if len(new_column_names) != len(set(new_column_names)):
                duplicates = [n for n in new_column_names if new_column_names.count(n) > 1]
                raise ValueError(
                    f"new_column_names contains duplicate names: {sorted(set(duplicates))}"
                )
        return new_column_names

    @staticmethod
    def _default_name(num_cols: list[str], denom_cols: list[str]) -> str:
        num_part = "__".join(num_cols) if num_cols else "const"
        denom_part = "__".join(denom_cols)
        return f"{num_part}__gratio__{denom_part}"

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "GeneralizedRatioFeatures":
        """Fit the transformer by resolving coefficients and column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        GeneralizedRatioFeatures
            Fitted transformer instance.
        """
        if self.numerator_coefficients is None:
            self.numerator_coefficients = [[1.0] * len(cols) for cols in self.numerator_columns]
        if self.denominator_coefficients is None:
            self.denominator_coefficients = [[1.0] * len(cols) for cols in self.denominator_columns]
        if self.numerator_biases is None:
            self.numerator_biases = [0.0] * len(self.numerator_columns)

        if self.new_column_names is None:
            default_names = [
                self._default_name(num_cols, denom_cols)
                for num_cols, denom_cols in zip(self.numerator_columns, self.denominator_columns, strict=False)
            ]

            if len(default_names) != len(set(default_names)):
                duplicates = [n for n in default_names if default_names.count(n) > 1]
                raise ValueError(
                    f"Duplicate (numerator, denominator) column groups produce identical "
                    f"auto-generated names: {sorted(set(duplicates))}. "
                    f"Use new_column_names to disambiguate."
                )
            self.new_column_names = default_names

        existing = set(X.columns)
        clashes = [name for name in self.new_column_names if name in existing]
        if clashes:
            raise ValueError(
                f"Output column names already exist in the input DataFrame: {clashes}. "
                f"Use new_column_names to choose different names."
            )

        self._output_dtypes = {col: pl.Float64 for col in self.new_column_names}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating generalized ratio features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with generalized ratio features appended.
        """
        new_columns = []

        assert (
            self.numerator_coefficients is not None
            and self.denominator_coefficients is not None
            and self.numerator_biases is not None
            and self.new_column_names is not None
        )
        for num_cols, denom_cols, num_coeffs, denom_coeffs, bias, new_col_name in zip(
            self.numerator_columns,
            self.denominator_columns,
            self.numerator_coefficients,
            self.denominator_coefficients,
            self.numerator_biases,
            self.new_column_names, strict=False,
        ):

            denom_terms = [
                pl.col(col).cast(pl.Float64) * c for col, c in zip(denom_cols, denom_coeffs, strict=False)
            ]
            denominator_expr = pl.sum_horizontal(denom_terms) + self.epsilon

            if num_cols:
                num_terms = [
                    pl.col(col).cast(pl.Float64) * c for col, c in zip(num_cols, num_coeffs, strict=False)
                ]
                numerator_expr = pl.sum_horizontal(num_terms) + bias
            else:
                numerator_expr = pl.lit(bias).cast(pl.Float64)

            new_columns.append((numerator_expr / denominator_expr).alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            columns_to_drop = list(
                {
                    col
                    for groups in (self.numerator_columns, self.denominator_columns)
                    for group in groups
                    for col in group
                }
            )
            X = X.drop(columns_to_drop)

        return X
