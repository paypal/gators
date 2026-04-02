from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin


class Endswith(BaseModel, BaseEstimator, TransformerMixin):
    """
    Generates Boolean features indicating if substrings are
    at the end of the original column values.

    Examples
    --------
    Create an instance of the Endswith class:

    >>> from gators.feature_feneration_str import Endswith
    >>> endswith_dict = {'col1': ['end1', 'end2'], 'col2': ['end3']}
    >>> transformer = Endswith(endswith_dict=endswith_dict)

    Fit the transformer:

    >>> transformer.fit(X)

    Transform the dataframe:

    >>> X = pl.DataFrame({"col1": ["this end1", None, "that end2"],
    ...                    "col2": [None, "one end3", "another no end"]})
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (3, 5)
    ╭─────────────┬───────────────┬─────────────────────┬─────────────────────┬─────────────────────╮
    │ col1        │ col2          │ col1__endswith_end1 │ col1__endswith_end2 │ col2__endswith_end3 │
    ├─────────────┼───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ this end1   │ None          │ True                │ False               │ None                │
    ├─────────────┼───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ None        │ one end3      │ None                │ None                │ True                │
    ├─────────────┼───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │ that end2   │ another no end│ False               │ True                │                False│
    ╰─────────────┴───────────────┴─────────────────────┴─────────────────────┴─────────────────────╯
    """

    endswith_dict: Dict[str, List[str]]

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Endswith":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Endswith
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
            pl.col(col).str.ends_with(substring).alias(f"{col}__endswith_{substring}")
            for col, substrings in self.endswith_dict.items()
            for substring in substrings
        ]
        return X.with_columns(transformations)
