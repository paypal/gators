from typing import Dict, List, Optional

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from pydantic import PositiveInt, field_validator

from ._base_discretizer import _BaseDiscretizer, generate_labels


class TreeBasedDiscretizer(_BaseDiscretizer):
    """
    Supervised discretizer using decision tree splits for optimal bin boundaries.

    Finds bin boundaries that maximize information gain or reduce variance by
    using decision tree split points. This is particularly effective for tree-based
    models as bins align with natural decision boundaries in the data.

    Parameters
    ----------
    subset : Optional[List[str]], default=None
        List of numeric column names to discretize. If None, all numeric columns are selected.
    num_bins : PositiveInt, default=5
        Maximum number of bins to create. Actual number may be less if tree finds fewer optimal splits.
    rounding : PositiveInt, default=3
        Decimal places to round bin edges for labels.
    inplace : bool, default=True
        If True, replace original columns with discretized values.
        If False, create new columns with suffix '__dic_tree'.
    drop_columns : bool, default=True
        If inplace=False, whether to drop the original columns after discretizing.
        Ignored when inplace=True.
    as_numerics : bool, default=False
        If True, create numeric labels (0, 1, 2, ...) instead of interval strings.
    task : str, default='classification'
        Type of supervised learning task: 'classification' or 'regression'.
        Determines which tree algorithm to use (LGBMClassifier vs LGBMRegressor).
    min_samples_leaf : int, default=10
        Minimum number of samples required in each leaf node (min_data_in_leaf in LightGBM).
        Controls the granularity of binning (higher = fewer, coarser bins).
    random_state : Optional[int], default=None
        Random state for reproducibility of tree splits.

    Examples
    --------
    **Example 1: Classification task**

    >>> from gators.discretizers import TreeBasedDiscretizer
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    ...     'income': [30000, 35000, 40000, 50000, 55000, 60000, 70000, 75000, 80000, 90000],
    ...     'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ... })
    >>> discretizer = TreeBasedDiscretizer(
    ...     subset=['age', 'income'],
    ...     num_bins=3,
    ...     task='classification',
    ...     drop_columns=True
    ... )
    >>> discretizer.fit(X, target='target')
    >>> transformed = discretizer.transform(X)
    >>> print(transformed)
    shape: (10, 3)
    ┌──────────────┬────────────────┬────────┐
    │ age__dic_tre ┆ income__dic_tr ┆ target │
    │ e            ┆ ee             ┆ ---    │
    │ ---          ┆ ---            ┆ i64    │
    │ str          ┆ str            ┆        │
    ├──────────────┼────────────────┼────────┤
    │ (-inf,42.5]  ┆ (-inf,52500.0] ┆ 0      │
    │ ...          ┆ ...            ┆ ...    │
    └──────────────┴────────────────┴────────┘

    **Example 2: Regression task**

    >>> X_reg = pl.DataFrame({
    ...     'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     'target': [10, 15, 18, 25, 30, 35, 42, 50]
    ... })
    >>> discretizer_reg = TreeBasedDiscretizer(
    ...     subset=['feature1'],
    ...     num_bins=4,
    ...     task='regression'
    ... )
    >>> discretizer_reg.fit(X_reg, target='target')
    >>> transformed = discretizer_reg.transform(X_reg)
    """

    task: str = "classification"
    min_samples_leaf: PositiveInt = 10
    random_state: Optional[int] = None

    @field_validator("task")
    def check_task(cls, task):
        if task not in ["classification", "regression"]:
            raise ValueError("task must be 'classification' or 'regression'")
        return task

    @field_validator("min_samples_leaf")
    def check_min_samples_leaf(cls, min_samples_leaf):
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        return min_samples_leaf

    def fit(self, X: pl.DataFrame, y: pl.Series = None) -> "TreeBasedDiscretizer":
        """Fit the discretizer by learning optimal splits from decision tree.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with numeric columns.
        y : pl.Series
            Target values (binary for classification, continuous for regression).
            Required for TreeBasedDiscretizer.

        Returns
        -------
        TreeBasedDiscretizer
            The fitted discretizer instance.

        Raises
        ------
        ValueError
            If y is None.
        """
        if y is None:
            raise ValueError(
                "TreeBasedDiscretizer requires a target variable 'y' for fitting. "
                "Please provide y when calling fit() or fit_transform(), e.g., "
                "discretizer.fit(X, y=y_train) or pipeline.fit_transform(X, y=y_train)"
            )

        # Identify numeric columns if not specified
        if not self.subset:
            self.subset = [
                col
                for col, dtype in dict(zip(X.columns, X.dtypes)).items()
                if dtype
                in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
            ]

        # Learn bins for each column using decision tree
        self._bins = {}
        for col in self.subset:

            # Create LightGBM model
            if self.task == "classification":
                tree = LGBMClassifier(
                    max_depth=int(np.log2(self.num_bins)) + 1,
                    min_data_in_leaf=self.min_samples_leaf,
                    n_estimators=1,
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                tree = LGBMRegressor(
                    max_depth=int(np.log2(self.num_bins)) + 1,
                    min_data_in_leaf=self.min_samples_leaf,
                    n_estimators=1,
                    random_state=self.random_state,
                    verbose=-1,
                )

            # Fit tree on single column
            X_col = X.select(pl.col(col)).to_numpy()
            tree.fit(X_col, y.to_numpy() if hasattr(y, 'to_numpy') else y)

            # Extract split thresholds from LightGBM tree
            thresholds = []
            tree_X = tree.booster_.trees_to_dataframe()
            
            # Get all split nodes (not leaf nodes)
            split_nodes = tree_X[tree_X['split_gain'].notna()]
            
            # Extract threshold values from split nodes
            if not split_nodes.empty:
                thresholds = split_nodes['threshold'].dropna().tolist()

            # Sort and deduplicate thresholds
            if thresholds:
                thresholds = sorted(set(thresholds))
                self._bins[col] = thresholds
            else:
                # If no splits found, use min/max
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                if min_val != max_val:
                    self._bins[col] = [(min_val + max_val) / 2]
                else:
                    self._bins[col] = []

        # Generate labels
        self._labels = generate_labels(self._bins, self.rounding)

        # Create column mapping
        self._column_mapping = {col: f"{col}__dic_tree" for col in self.subset}

        return self
