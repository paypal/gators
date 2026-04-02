import polars as pl

from ..encoders.woe_encoder import compute_woe_iv


def compute_iv(X, y, regularization=0.01):
    """
    Compute the Information Value (IV) for each categorical feature in the dataset.

    To convert continuous features to categorical, consider using the `binning` module to create bins before computing IV.

    Parameters
    ----------
    X : pl.DataFrame
        The input features.
    y : pl.Series
        The target variable (binary).
    regularization : float, default=0.01
        Regularization parameter to avoid division by zero in WOE/IV calculation.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the IV values for each feature.

    Examples
    --------
    >>> import polars as pl
    >>> from gators.feature_selection import compute_iv

    >>> X = pl.DataFrame({
    ...     "feature1": ["a", "a", "b", "c"],
    ...     "feature2": ["x", "x", "x", "y"],
    ...     "target": [1, 0, 1, 0]
    ... })
    >>> iv = compute_iv(X.drop("target"), X["target"])
    >>> print(iv)
    shape: (2, 2)
    ┌──────────┬────────────┐
    │ feature  │ iv         │
    │ ---      │ ---        │
    │ str      │ f64        │
    ╞══════════╪════════════╡
    │ feature1 │ 0.693147   │
    │ feature2 │ 0.287682   │
    └──────────┴────────────┘
    """
    string_cat_cols = [
        col for col, dtype in X.schema.items() if dtype in [pl.String, pl.Categorical]
    ]

    # Return empty DataFrame if no String/Categorical columns
    if not string_cat_cols:
        return pl.DataFrame(
            {"feature": [], "iv": []}, schema={"feature": pl.String, "iv": pl.Float64}
        )

    X_filtered = X.select(string_cat_cols)

    # Compute WOE/IV statistics and aggregate IV by variable
    woe_iv_stats = compute_woe_iv(X_filtered, y, regularization=regularization)
    return woe_iv_stats.group_by("variable").agg(pl.col("iv").sum()).rename({"variable": "feature"})
