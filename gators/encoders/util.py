from typing import Tuple

import polars as pl


def determine_encoding_strategy(train: pl.DataFrame, max_count_woe: int = 100) -> Tuple[list, list]:
    """
    Determine which categorical columns should use WOE vs. one-hot encoding based on cardinality.

    Parameters
    ----------
    train : pl.DataFrame
        Training dataset containing categorical features
    max_count_woe : int, default=100
        Maximum unique value threshold for WOE encoding. Columns with cardinality
        <= threshold use WOE encoding, others use one-hot encoding

    Returns
    -------
    Tuple[list, list]
        - woe_columns: List of column names for WOE encoding (low cardinality)
        - onehot_columns: List of column names for one-hot encoding (high cardinality)
    """
    string_columns = [
        col
        for col, dtype in zip(train.columns, train.dtypes)
        if dtype in [pl.Enum, pl.String]
    ]

    column_counts = {col: train[col].n_unique() for col in string_columns}
    woe_columns = [col for col, count in column_counts.items() if count <= max_count_woe]
    onehot_columns = [col for col, count in column_counts.items() if count > max_count_woe]
    return woe_columns, onehot_columns
