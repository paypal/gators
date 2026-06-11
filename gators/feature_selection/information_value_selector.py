import polars as pl
from pydantic import PrivateAttr

from ..transformer._base_transformer import _BaseTransformer
from .information_value import compute_iv


class InformationValueSelector(_BaseTransformer):
    """Drop columns whose Information Value falls below a threshold.

    Computes the IV for each categorical (String/Categorical/Enum) column
    and drops those whose IV is below ``threshold``.  Numeric columns are
    always kept untouched; they are not considered for IV computation.

    Parameters
    ----------
    threshold : float, default=0.02
        Minimum IV required to keep a column.  Columns with IV strictly
        below this value are dropped.
    regularization : float, default=0.01
        Regularization applied to WOE/IV calculation to avoid division by zero.

    Attributes
    ----------
    selected_features_ : list[str]
        All column names that survive the threshold (set after ``fit``).
    columns_to_drop_ : list[str]
        Categorical column names dropped because their IV was too low.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_selection import InformationValueSelector

    >>> X = pl.DataFrame({
    ...     "cat_strong": ["a", "b", "a", "b", "a", "b"],
    ...     "cat_weak":   ["x", "x", "x", "x", "y", "y"],
    ...     "numeric":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ... })
    >>> y = pl.Series("target", [1, 0, 1, 0, 1, 0])
    >>> selector = InformationValueSelector(threshold=0.02)
    >>> selector.fit(X, y)
    >>> X_transformed = selector.transform(X)
    """

    threshold: float = 0.02
    regularization: float = 0.01

    _selected_features: list[str] = PrivateAttr(default_factory=list)
    _columns_to_drop: list[str] = PrivateAttr(default_factory=list)

    @property
    def selected_features_(self) -> list[str]:
        return self._selected_features

    @property
    def columns_to_drop_(self) -> list[str]:
        return self._columns_to_drop

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> "InformationValueSelector":
        """Compute IV for each categorical column and record which to drop.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : pl.Series
            Binary target series.

        Returns
        -------
        InformationValueSelector
            The fitted transformer instance.
        """
        if y is None:
            raise ValueError("y must be provided for InformationValueSelector.fit()")

        iv_df = compute_iv(X, y, regularization=self.regularization)

        below_threshold = set(iv_df.filter(pl.col("iv") < self.threshold)["feature"].to_list())

        self._columns_to_drop = [col for col in X.columns if col in below_threshold]
        self._selected_features = [col for col in X.columns if col not in below_threshold]
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Drop low-IV columns from the DataFrame.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            DataFrame with low-IV categorical columns removed.
        """
        if not self._columns_to_drop:
            return X
        return X.drop(self._columns_to_drop)
