from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class Startswith(_BaseTransformer):
    """
    Generates Boolean features to indicate if strings in the
    original columns start with specified substrings.

    Examples
    --------
    Create an instance of the Startswith class:

    >>> from gators.feature_feneration_str import Startswith
    >>> startswith_dict = {'col1': ['pre1', 'pre2'], 'col2': ['pre3']}
    >>> transformer = Startswith(startswith_dict=startswith_dict)

    Fit the transformer:

    >>> transformer.fit(X)

    Transform the DataFrame:

    >>> X = pl.DataFrame({"col1": ["pre1_sample", None, "pre2_sample"],
    ...                     "col2": [None, "pre3_sample", "no_match"]})
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (3, 5)
    ┌───────────────┬─────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
    │ col1          │ col2        │ col1__startswith_pre1 │ col1__startswith_pre2 │ col2__startswith_pre3 │
    ├───────────────┼─────────────┼───────────────────────┼───────────────────────┼───────────────────────┤
    │ pre1_sample   │ None        │ True                  │ False                 │ None                  │
    ├───────────────┼─────────────┼───────────────────────┼───────────────────────┼───────────────────────┤
    │ None          │ pre3_sample │ None                  │ None                  │ True                  │
    ├───────────────┼─────────────┼───────────────────────┼───────────────────────┼───────────────────────┤
    │ pre2_sample   │ no_match    │ False                 │ True                  │ False                 │
    └───────────────┴─────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
    """

    startswith_dict: Dict[str, List[str]]

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Startswith":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Startswith
            Fitted transformer instance.
        """
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
        transformations = [
            pl.col(col).str.starts_with(substring).alias(f"{col}__startswith_{substring}")
            for col, substrings in self.startswith_dict.items()
            for substring in substrings
        ]
        return X.with_columns(*transformations)
