import re

import polars as pl


def map_substring_replacements(X: pl.DataFrame, old: str, new: str) -> dict:
    """
    Create a mapping to replace substrings in string column values.

    This function is primarily used to handle column names containing whitespaces
    that are generated after one-hot encoding. LightGBM (and some other ML libraries)
    throw errors when column names contain whitespaces, so this utility helps create
    a mapping to replace them (e.g., spaces with underscores).

    Parameters
    ----------
    X : pl.DataFrame
        Input DataFrame with string columns to process.
    old : str
        Substring to search for in the string values (e.g., " " for whitespace).
    new : str
        Replacement substring (e.g., "_" for underscore).

    Returns
    -------
    dict
        Nested dictionary mapping column names to value replacements:
        {column_name: {old_value: new_value, ...}, ...}

    Examples
    --------
    >>> import polars as pl
    >>> X = pl.DataFrame({
    ...     'categories': ['cat A', 'cat B', 'cat A', 'dog C'],
    ...     'numbers': [1, 2, 3, 4]
    ... })
    >>> # Replace spaces with underscores in string values
    >>> mapping = map_substring_replacements(X, old=' ', new='_')
    >>> print(mapping)
    {'categories': {'cat A': 'cat_A', 'cat B': 'cat_B', 'dog C': 'dog_C'}}

    Notes
    -----
    - Only processes string and categorical columns (pl.String or pl.Categorical dtype)
    - Only includes columns where the substring is found
    - Common use case: After one-hot encoding, column names like "color__light blue"
      should be changed to "color__light_blue" for LightGBM compatibility
    """
    mapping = {}
    # Escape special regex characters in the old substring
    escaped_old = re.escape(old)

    for column in X.columns:
        if X[column].dtype not in [pl.String, pl.Categorical]:
            continue

        # Cast categorical to string for str operations
        col_expr = X[column].cast(pl.String) if X[column].dtype == pl.Categorical else X[column]

        filtered_series = col_expr.filter(col_expr.str.contains(escaped_old))
        categories = filtered_series.unique()
        if categories.is_empty():
            continue
        new_categories = [cat.replace(old, new) for cat in categories]
        mapping[column] = dict(zip(categories, new_categories))

    return mapping
