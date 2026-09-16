import polars as pl


def determine_encoding_strategy(X: pl.DataFrame, max_count_woe: int = 100) -> tuple[list, list]:
    """
    Determine which categorical columns should use WOE vs. one-hot encoding based on cardinality.

    Parameters
    ----------
    X : pl.DataFrame
        Training dataset containing categorical features
    max_count_woe : int, default=100
        Maximum unique value threshold for WOE encoding. Columns with cardinality
        <= threshold use WOE encoding, others use one-hot encoding

    Returns
    -------
    tuple[ list, list]
        - woe_columns: List of column names for WOE encoding (low cardinality)
        - onehot_columns: List of column names for one-hot encoding (high cardinality)
    """
    _CAT_DTYPES = {pl.String, pl.Categorical, pl.Enum}
    string_columns = [
        col for col, dtype in zip(X.columns, X.dtypes, strict=False) if dtype.base_type() in _CAT_DTYPES
    ]
    column_counts = {col: X[col].n_unique() for col in string_columns}
    woe_columns = [col for col, count in column_counts.items() if count <= max_count_woe]
    onehot_columns = [col for col, count in column_counts.items() if count > max_count_woe]
    return woe_columns, onehot_columns
