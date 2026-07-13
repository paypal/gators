import polars as pl
from pydantic import field_validator

from ..transformer._base_transformer import _BaseTransformer

OPERATION_FUNCTIONS = {
    "sum": lambda cols: pl.sum_horizontal(*cols),
    "mean": lambda cols: pl.mean_horizontal(*cols),
    "minus": lambda cols: pl.reduce(lambda x, y: x - y, cols),
    "mul": lambda cols: pl.reduce(lambda x, y: x * y, cols),
    "div": lambda cols: pl.reduce(lambda x, y: x / y, cols),
    "min": lambda cols: pl.min_horizontal(*cols),
    "max": lambda cols: pl.max_horizontal(*cols),
    "std": lambda cols: pl.concat_list(*cols).list.eval(pl.element().std()).list.first(),
    "var": lambda cols: pl.concat_list(*cols).list.eval(pl.element().var()).list.first(),
    "median": lambda cols: pl.concat_list(*cols).list.eval(pl.element().median()).list.first(),
    "range": lambda cols: pl.max_horizontal(*cols) - pl.min_horizontal(*cols),
    "abs_diff": lambda cols: pl.reduce(lambda x, y: (x - y).abs(), cols),
    "count_null": lambda cols: pl.concat_list(*cols).list.eval(pl.element().is_null().sum()),
    "count_zero": lambda cols: pl.concat_list(*cols).list.eval((pl.element() == 0).sum()),
    "count_nonzero": lambda cols: pl.concat_list(*cols).list.eval((pl.element() != 0).sum()),
}


class MathFeatures(_BaseTransformer):
    """
    Generates new features by applying mathematical operations to groups of columns.

    Parameters
    ----------
    groups : list[list[str]]
        List of groups of column names to apply operations on.
    func : list[str]
        List of operations to apply to each group of columns. Available operations:

        - 'sum': Sum of all columns
        - 'mean': Mean of all columns
        - 'minus': Subtraction (reduces columns left to right)
        - 'mul': Product of all columns
        - 'div': Division (reduces columns left to right)
        - 'min': Minimum value across columns
        - 'max': Maximum value across columns
        - 'std': Standard deviation across columns
        - 'var': Variance across columns
        - 'median': Median across columns
        - 'range': Range (max - min)
        - 'abs_diff': Absolute difference (reduces columns left to right)
        - 'count_null': Count of null values
        - 'count_zero': Count of zero values
        - 'count_nonzero': Count of non-zero values

        Note: For division operations, consider using RatioFeatures instead, which provides safer
        division with automatic handling of division by zero and null values.
    drop_columns : bool, optional
        Whether to drop the original columns after creating the new features, by default False.
    new_column_names : list[str]], optional
        List of new column names for the created features, by default None.

    Examples
    --------
    >>> from math_features import MathFeatures
    >>> import polars as pl

    >>> X ={'A': [1, 2, 3, 4],
    ...         'B': [4, 3, 2, 1],
    ...         'C': [1, 2, 1, 2]}
    >>> X = pl.DataFrame(X)

    **Example 1: drop_columns=False**

    >>> transformer = MathFeatures(groups=[['A', 'B'], ['B', 'C']], func=['sum', 'mean'])
    >>> transformer.fit(X)
    MathFeatures(groups=[['A', 'B'], ['B', 'C']], func=['sum', 'mean'])
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 6)
    ┌─────┬─────┬─────┬────────┬─────-───┬────────┐
    │  A  │  B  │  C  │ A_B_sum│ A_B_mean│ B_C_sum│
    │ i64 │ i64 │ i64 │  f64   │  f64    │  f64   │
    ├─────┼─────┼─────┼────────┼──────-──┼────────┤
    │  1  │  4  │  1  │  5.0   │  2.5    │  5.0   │
    │  2  │  3  │  2  │  5.0   │  2.5    │  5.0   │
    │  3  │  2  │  1  │  5.0   │  2.5    │  3.0   │
    │  4  │  1  │  2  │  5.0   │  2.5    │  3.0   │
    └─────┴─────┴─────┴────────┴───────-─┴────────┘

    **Example 2: drop_columns=True**

    >>> transformer = MathFeatures(groups=[['A', 'B'], ['B', 'C']], func=['sum'], drop_columns=True)
    >>> transformer.fit(X)
    MathFeatures(groups=[['A', 'B'], ['B', 'C']], func=['sum'], drop_columns=True)
    >>> result = transformer.transform(X)
    >>> result
    shape: (4, 2)
    ┌────────┬────────┐
    │ A_B_sum│ B_C_sum│
    │  f64   │  f64   │
    ├────────┼────────┤
    │  5.0   │  5.0   │
    │  5.0   │  5.0   │
    │  5.0   │  3.0   │
    │  5.0   │  3.0   │
    └────────┴────────┘
    """

    groups: list[list[str]]
    func: list[str]
    drop_columns: bool = False
    new_column_names: list[str] | None = None
    _column_mapping: dict[str, str] = {}

    @field_validator("func")
    def check_func(cls, func):
        for f in func:
            if f not in list(OPERATION_FUNCTIONS.keys()):
                raise ValueError(f"{f} is not in the predefined list of datetime functions.")
        return func

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "MathFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series, default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        MathFeatures
            Fitted transformer instance.
        """
        default_names = [f"{'_'.join(group)}" for group in self.groups]
        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by extracting specified components.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        new_columns = []
        for group in self.groups:
            name = f"{'_'.join(group)}"
            for op in self.func:
                new = f"{self._column_mapping[name]}_{op}"
                new_columns.append(
                    OPERATION_FUNCTIONS[op]([pl.col(col) for col in group]).alias(new)
                )
        X = X.with_columns(new_columns)

        if self.drop_columns:
            return X.drop([col for group in self.groups for col in group])

        return X
