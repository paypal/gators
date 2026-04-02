from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel, PrivateAttr, field_validator
from sklearn.base import BaseEstimator, TransformerMixin

AGGREGATION_FUNCTIONS = ["mean", "std", "median", "min", "max", "sum", "count"]


def validate_and_convert_window(window_str: str) -> str:
    """
    Validate window string and convert to polars-compatible format.

    Supported input formats:

    - '30m' -> '30m' (30 minutes)
    - '1h' -> '1h' (1 hour)
    - '24h' -> '24h' (24 hours)
    - '7d' -> '7d' (7 days)
    - '1M' -> '30d' (1 month approximated as 30 days)
    - '1Y' -> '365d' (1 year approximated as 365 days)

    Returns polars-compatible window string.
    """
    import re

    match = re.match(r"^(\d+)([mhdMY])$", window_str)
    if not match:
        raise ValueError(
            f"Invalid window format: {window_str}. "
            f"Expected format: number followed by m/h/d/M/Y (e.g., '1h', '24h', '7d')"
        )

    value = int(match.group(1))
    unit = match.group(2)

    # Convert custom units to polars-compatible format
    if unit == "M":
        return f"{value * 30}d"  # Convert months to days
    elif unit == "Y":
        return f"{value * 365}d"  # Convert years to days
    else:
        # m, h, d are already compatible
        return window_str


class TimeWindowFeatures(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates time-based window aggregation features (velocity features).

    This transformer creates rolling window statistics over time periods, useful for
    detecting unusual patterns like transaction velocity, spending bursts, etc.

    Features are computed looking backward from each row (excluding the current row),
    optionally grouped by categorical columns.

    Parameters
    ----------
    subset : List[str]
        List of numerical column names to aggregate over time windows.
    time_column : str
        Name of the datetime column to use for time-based windowing.
    windows : List[str]
        List of time window strings. Supported formats:

        - '30m' = 30 minutes
        - '1h' = 1 hour
        - '24h' = 24 hours
        - '7d' = 7 days
        - '1M' = 1 month (30 days)
        - '1Y' = 1 year (365 days)
    by : Optional[List[str]], default=None
        Optional list of columns to group by. Windows are computed within each group.
        Example: ['card1'] computes "transactions in last 24h for this card"
    func : List[str], default=['count', 'mean']
        List of aggregation functions to apply. Available options:

        - 'count': Count of rows in window
        - 'mean': Mean of values in window
        - 'sum': Sum of values in window
        - 'std': Standard deviation in window
        - 'median': Median in window
        - 'min': Minimum value in window
        - 'max': Maximum value in window
    drop_columns : bool, default=False
        Whether to drop the original numerical columns after creating features.
    new_column_names : Optional[List[str]], default=None
        List of custom names for the window columns. If None, uses default naming pattern.

    Examples
    --------
    >>> from gators.feature_generation import TimeWindowFeatures
    >>> import polars as pl
    >>> from datetime import datetime, timedelta

    >>> # Sample transaction data
    >>> X ={
    ...     'TransactionDT': [
    ...         datetime(2024, 1, 1, 10, 0),
    ...         datetime(2024, 1, 1, 10, 30),
    ...         datetime(2024, 1, 1, 11, 0),
    ...         datetime(2024, 1, 1, 12, 0),
    ...         datetime(2024, 1, 2, 10, 0),
    ...     ],
    ...     'TransactionAmt': [100, 150, 200, 120, 180],
    ...     'card1': ['A', 'A', 'A', 'B', 'A']
    ... }
    >>> X = pl.DataFrame(X)

    **Example 1: Global time windows (no grouping)**

    >>> transformer = TimeWindowFeatures(
    ...     subset=['TransactionAmt'],
    ...     time_column='TransactionDT',
    ...     windows=['1h', '24h'],
    ...     func=['count', 'mean']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> result.columns
    ['TransactionDT', 'TransactionAmt', 'card1',
     'count_TransactionAmt_1h', 'mean_TransactionAmt_1h',
     'count_TransactionAmt_24h', 'mean_TransactionAmt_24h']

    **Example 2: Grouped time windows (per card)**

    >>> transformer = TimeWindowFeatures(
    ...     subset=['TransactionAmt'],
    ...     time_column='TransactionDT',
    ...     windows=['1h', '24h'],
    ...     by=['card1'],
    ...     func=['count', 'sum']
    ... )
    >>> result = transformer.fit_transform(X)
    >>> # Creates: count/sum of TransactionAmt in last 1h/24h per card1

    **Example 3: Multiple windows for fraud detection**

    >>> transformer = TimeWindowFeatures(
    ...     subset=['TransactionAmt'],
    ...     time_column='TransactionDT',
    ...     windows=['30m', '1h', '3h', '24h', '7d'],
    ...     by=['card1'],
    ...     func=['count', 'mean', 'sum', 'max']
    ... )
    >>> # Detects velocity: "Card has 50 transactions in last hour"

    Notes
    -----

    - Data should be sorted by time_column for correct window calculations
    - Current row is EXCLUDED from window calculations
    - First rows (no history) have null values
    - Windows look backward from current time
    - Useful for velocity features, spending patterns, anomaly detection
    """

    subset: List[str]
    time_column: str
    windows: List[str]
    by: Optional[List[str]] = None
    func: List[str] = ["count", "mean"]
    drop_columns: bool = False
    new_column_names: Optional[List[str]] = None
    _column_mapping: Dict[str, str] = PrivateAttr(default_factory=dict)
    _converted_windows: Dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("windows")
    def check_windows(cls, windows):
        for window in windows:
            try:
                validate_and_convert_window(window)
            except ValueError as e:
                raise ValueError(f"Invalid window format: {e}")
        return windows

    @field_validator("func")
    def check_aggregations(cls, func):
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
            windows = info.data.get("windows", [])
            func = info.data.get("func", [])
            expected_length = len(subset) * len(windows) * len(func)
            if len(new_column_names) != expected_length:
                raise ValueError(
                    f"Length of new_column_names ({len(new_column_names)}) "
                    f"must match the total number of features created ({expected_length})"
                )
        return new_column_names

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "TimeWindowFeatures":
        """Fit the transformer by generating column name mappings.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        TimeWindowFeatures
            Fitted transformer instance.
        """
        # Convert windows to polars-compatible format once during fit
        self._converted_windows = {w: validate_and_convert_window(w) for w in self.windows}

        # Generate default column names
        default_names = []
        group_suffix = f"_{'_'.join(self.by)}" if self.by else ""

        for num_col in self.subset:
            for window in self.windows:
                for fun in self.func:
                    default_names.append(f"{fun}_{num_col}_{window}{group_suffix}")

        if not self.new_column_names:
            self.new_column_names = default_names
        self._column_mapping = dict(zip(default_names, self.new_column_names))

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame by creating time window features.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame to transform. Should be sorted by time_column.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame with time window features.
        """
        new_columns = []
        group_suffix = f"_{'_'.join(self.by)}" if self.by else ""

        for num_col in self.subset:
            for window_str, window_polars in self._converted_windows.items():
                for fun in self.func:
                    default_name = f"{fun}_{num_col}_{window_str}{group_suffix}"
                    new_col_name = self._column_mapping[default_name]

                    # Use rolling with group_by for time-based windows
                    # rolling_* excludes current row by default
                    over_cols = self.by if self.by else []

                    if fun == "count":
                        # For count, use any column (time column is guaranteed to exist)
                        if over_cols:
                            expr = (
                                pl.col(self.time_column)
                                .count()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",  # Exclude current row
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(self.time_column)
                                .count()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )
                    elif fun == "mean":
                        if over_cols:
                            expr = (
                                pl.col(num_col)
                                .mean()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(num_col)
                                .mean()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )
                    elif fun == "sum":
                        if over_cols:
                            expr = (
                                pl.col(num_col)
                                .sum()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(num_col)
                                .sum()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )
                    elif fun == "std":
                        if over_cols:
                            expr = (
                                pl.col(num_col)
                                .std()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(num_col)
                                .std()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )
                    elif fun == "median":
                        if over_cols:
                            expr = (
                                pl.col(num_col)
                                .median()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(num_col)
                                .median()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )
                    elif fun == "min":
                        if over_cols:
                            expr = (
                                pl.col(num_col)
                                .min()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(num_col)
                                .min()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )
                    elif fun == "max":
                        if over_cols:
                            expr = (
                                pl.col(num_col)
                                .max()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                                .over(over_cols)
                            )
                        else:
                            expr = (
                                pl.col(num_col)
                                .max()
                                .rolling(
                                    index_column=self.time_column,
                                    period=window_polars,
                                    closed="left",
                                )
                            )

                    new_columns.append(expr.alias(new_col_name))

        X = X.with_columns(new_columns)

        if self.drop_columns:
            X = X.drop(self.subset)

        return X
