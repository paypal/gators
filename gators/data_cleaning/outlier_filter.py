from typing import Annotated, Dict, List, Literal, Optional

import polars as pl
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierFilter(BaseModel, BaseEstimator, TransformerMixin):
    """
    Removes or caps outliers in numerical columns using various methods.

    Detects outliers using IQR, Z-score, or percentile methods and either
    removes rows or caps values. Essential for tree-based models to prevent
    splits dominated by extreme values.

    Supports class-aware outlier detection for imbalanced datasets to avoid
    removing minority class examples that appear as statistical outliers.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric columns to check for outliers. If None, all numeric
        columns are checked.
    method : str, default='iqr'

        Method for outlier detection:

        - 'iqr': Interquartile Range method (values outside Q1-k*IQR, Q3+k*IQR)
        - 'zscore': Z-score method (values with absolute z-score > threshold)
        - 'percentile': Percentile method (values outside specified percentiles)
    threshold : float, default=1.5
        Threshold parameter for outlier detection:

        - For 'iqr': multiplier for IQR (typically 1.5 or 3.0)
        - For 'zscore': z-score threshold (typically 3.0)
        - Not used for 'percentile' method
    lower_percentile : float, default=0.01
        Lower percentile for outlier detection (only for 'percentile' method).
        Values below this percentile are considered outliers.
    upper_percentile : float, default=0.99
        Upper percentile for outlier detection (only for 'percentile' method).
        Values above this percentile are considered outliers.
    action : str, default='remove'
        Action to take on outliers:
        - 'remove': Remove rows containing outliers
        - 'cap': Cap outliers to boundary values
    class_aware : bool, default=False
        Whether to detect outliers separately within each class. Prevents
        removing minority class examples that appear as outliers when
        considering all data together. Requires passing target column name
        to fit(). Recommended for imbalanced classification tasks.

    Examples
    --------
    **Example 1: IQR method with row removal**

    >>> from gators.data_cleaning import OutlierFilter
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 200],  # 200 is outlier
    ...     'income': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
    ... })
    >>> filter_iqr = OutlierFilter(
    ...     subset=['age'],
    ...     method='iqr',
    ...     threshold=1.5,
    ...     action='remove'
    ... )
    >>> filter_iqr.fit(X)
    >>> result = filter_iqr.transform(X)
    >>> print(result)
    shape: (9, 2)
    ┌─────┬────────┐
    │ age ┆ income │
    │ --- ┆ ---    │
    │ i64 ┆ i64    │
    ├─────┼────────┤
    │ 25  ┆ 30000  │
    │ 30  ┆ 35000  │
    │ ... ┆ ...    │
    │ 65  ┆ 70000  │
    └─────┴────────┘

    **Example 2: Z-score method with capping**

    >>> filter_zscore = OutlierFilter(
    ...     subset=['age'],
    ...     method='zscore',
    ...     threshold=3.0,
    ...     action='cap'
    ... )
    >>> filter_zscore.fit(X)
    >>> result = filter_zscore.transform(X)

    **Example 3: Percentile method**

    >>> filter_percentile = OutlierFilter(
    ...     subset=['income'],
    ...     method='percentile',
    ...     lower_percentile=0.05,
    ...     upper_percentile=0.95,
    ...     action='remove'
    ... )
    >>> filter_percentile.fit(X)
    >>> result = filter_percentile.transform(X)

    **Example 4: Class-aware mode for imbalanced datasets**

    >>> X = pl.DataFrame({
    ...     'transaction_amount': [100, 120, 110, 105, 115, 5000, 4800, 4900],
    ...     'is_fraud': [0, 0, 0, 0, 0, 1, 1, 1]
    ... })
    >>> filter_basic = OutlierFilter(
    ...     subset=['transaction_amount'],
    ...     method='iqr',
    ...     action='remove',
    ...     class_aware=False
    ... )
    >>> filter_basic.fit(X)
    >>> result_basic = filter_basic.transform(X)
    >>> print(len(result_basic))  # May remove fraud examples!
    5
    >>> filter_aware = OutlierFilter(
    ...     subset=['transaction_amount'],
    ...     method='iqr',
    ...     action='remove',
    ...     class_aware=True
    ... )
    >>> filter_aware.fit(X, y='is_fraud')
    >>> result_aware = filter_aware.transform(X)
    >>> print(len(result_aware))  # Preserves minority class!
    8
    """

    subset: Optional[List[str]] = None
    method: Literal["iqr", "zscore", "percentile"] = "iqr"
    threshold: float = 1.5
    lower_percentile: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    upper_percentile: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.99
    action: Literal["remove", "cap"] = "remove"
    class_aware: bool = False
    _bounds: Dict = {}
    _target_col: Optional[str] = None

    def fit(self, X: pl.DataFrame, y: Optional[str] = None) -> "OutlierFilter":
        """Fit the transformer by computing outlier bounds.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[str], default=None
            Target column name (must exist in X). Required when class_aware=True.
            The column should contain class labels for class-aware outlier detection.
            Not used when class_aware=False (present for pipeline compatibility).

        Returns
        -------
        OutlierFilter
            Fitted transformer instance.

        Raises
        ------
        ValueError
            If class_aware=True and y is None.
            If y is provided but not found in X columns.
        """
        # Validate class_aware requirements
        if self.class_aware:
            if y is None:
                raise ValueError("Target column name 'y' must be provided when class_aware=True")
            if y not in X.columns:
                raise ValueError(f"Target column '{y}' not found in DataFrame")
            self._target_col = y

        # Identify numeric columns if not specified
        if not self.subset:
            self.subset = [
                col
                for col, dtype in dict(zip(X.columns, X.dtypes)).items()
                if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
            ]
            # Exclude target column from outlier detection
            if self.class_aware and self._target_col in self.subset:
                self.subset.remove(self._target_col)

        self._bounds = {}

        if self.class_aware:
            # Compute bounds separately for each class
            classes = X[self._target_col].unique().to_list()

            for col in self.subset:
                self._bounds[col] = {}

                for class_val in classes:
                    # Filter data for this class
                    class_data = X.filter(pl.col(self._target_col) == class_val)[col]

                    if self.method == "iqr":
                        q1 = class_data.quantile(0.25)
                        q3 = class_data.quantile(0.75)
                        iqr = q3 - q1 if q1 is not None and q3 is not None else 0
                        lower_bound = q1 - self.threshold * iqr if q1 is not None else None
                        upper_bound = q3 + self.threshold * iqr if q3 is not None else None

                    elif self.method == "zscore":
                        mean = class_data.mean()
                        std = class_data.std()
                        if mean is not None and std is not None:
                            lower_bound = mean - self.threshold * std
                            upper_bound = mean + self.threshold * std
                        else:
                            lower_bound = upper_bound = None

                    elif self.method == "percentile":
                        lower_bound = class_data.quantile(self.lower_percentile)
                        upper_bound = class_data.quantile(self.upper_percentile)

                    self._bounds[col][str(class_val)] = {
                        "lower": lower_bound,
                        "upper": upper_bound,
                    }
        else:
            # Compute bounds globally across all data
            for col in self.subset:
                if self.method == "iqr":
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1 if q1 is not None and q3 is not None else 0
                    lower_bound = q1 - self.threshold * iqr if q1 is not None else None
                    upper_bound = q3 + self.threshold * iqr if q3 is not None else None

                elif self.method == "zscore":
                    mean = X[col].mean()
                    std = X[col].std()
                    if mean is not None and std is not None:
                        lower_bound = mean - self.threshold * std
                        upper_bound = mean + self.threshold * std
                    else:
                        lower_bound = upper_bound = None

                elif self.method == "percentile":
                    lower_bound = X[col].quantile(self.lower_percentile)
                    upper_bound = X[col].quantile(self.upper_percentile)

                self._bounds[col] = {"lower": lower_bound, "upper": upper_bound}

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the DataFrame by handling outliers.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with outliers handled.
        """
        if not self.subset:
            return X

        if self.class_aware:
            # Class-aware mode: check outliers per class
            if self._target_col not in X.columns:
                raise ValueError(f"Target column '{self._target_col}' not found in transform data")

            if self.action == "remove":
                # Build filter condition per class
                filter_condition = pl.lit(False)

                for class_val, class_bounds in self._bounds[self.subset[0]].items():
                    class_filter = pl.col(self._target_col) == (
                        int(class_val) if class_val.isdigit() else class_val
                    )

                    for col in self.subset:
                        bounds = self._bounds[col][class_val]
                        if bounds["lower"] is not None and bounds["upper"] is not None:
                            class_filter = class_filter & (
                                (pl.col(col) >= bounds["lower"]) & (pl.col(col) <= bounds["upper"])
                            )

                    filter_condition = filter_condition | class_filter

                return X.filter(filter_condition)

            elif self.action == "cap":
                # Cap outliers per class
                transformations = []

                for col in self.subset:
                    capped_expr = None

                    for class_val, bounds in self._bounds[col].items():
                        class_val_typed = int(class_val) if class_val.isdigit() else class_val

                        if bounds["lower"] is not None and bounds["upper"] is not None:
                            class_capped = (
                                pl.when(pl.col(col) < bounds["lower"])
                                .then(bounds["lower"])
                                .when(pl.col(col) > bounds["upper"])
                                .then(bounds["upper"])
                                .otherwise(pl.col(col))
                            )

                            if capped_expr is None:
                                capped_expr = pl.when(
                                    pl.col(self._target_col) == class_val_typed
                                ).then(class_capped)
                            else:
                                capped_expr = capped_expr.when(
                                    pl.col(self._target_col) == class_val_typed
                                ).then(class_capped)

                    if capped_expr is not None:
                        capped_expr = capped_expr.otherwise(pl.col(col)).alias(col)
                        transformations.append(capped_expr)
                    else:
                        transformations.append(pl.col(col))

                # Keep other columns as is
                other_cols = [col for col in X.columns if col not in self.subset]
                for col in other_cols:
                    transformations.append(pl.col(col))

                return X.select(transformations)

        else:
            # Global mode: check outliers across all data
            if self.action == "remove":
                # Build filter condition for all columns
                filter_condition = pl.lit(True)
                for col in self.subset:
                    bounds = self._bounds[col]
                    if bounds["lower"] is not None and bounds["upper"] is not None:
                        filter_condition = filter_condition & (
                            (pl.col(col) >= bounds["lower"]) & (pl.col(col) <= bounds["upper"])
                        )

                return X.filter(filter_condition)

            elif self.action == "cap":
                # Cap outliers to boundary values
                transformations = []
                for col in self.subset:
                    bounds = self._bounds[col]
                    if bounds["lower"] is not None and bounds["upper"] is not None:
                        capped = (
                            pl.when(pl.col(col) < bounds["lower"])
                            .then(bounds["lower"])
                            .when(pl.col(col) > bounds["upper"])
                            .then(bounds["upper"])
                            .otherwise(pl.col(col))
                            .alias(col)
                        )
                        transformations.append(capped)
                    else:
                        transformations.append(pl.col(col))

                # Keep non-outlier columns as is
                other_cols = [col for col in X.columns if col not in self.subset]
                for col in other_cols:
                    transformations.append(pl.col(col))

                return X.select(transformations)

        return X
