from typing import Dict, List, Optional

import polars as pl

from ..transformer._base_transformer import _BaseTransformer


class Contains(_BaseTransformer):
    """
    Generates Boolean columns indicating if substrings are
    contained within the original column values.

    Examples
    --------
    Create an instance of the Contains class:

    >>> from gators.feature_generation_str import Contains
    >>> contains_dict = {'col1': ['sub1', 'sub2'], 'col2': ['sub3']}
    >>> transformer = Contains(contains_dict=contains_dict)

    Fit the transformer:

    >>> transformer.fit(X)

    Transform the dataframe:

    >>> X = pl.DataFrame({"col1": ["sub1 here", None, "sub2 here"],
    ...                    "col2": [None, "sub3 inside", "no match"]})
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    shape: (3, 5)
    ╭─────────────┬───────────────┬──────────────┬──────────────┬──────────────╮
    │ col1        │ col2          │ col1__sub1   │ col1__sub2   │ col2__sub3   │
    ├─────────────┼───────────────┼──────────────┼──────────────┼──────────────┤
    │ sub1 here   │ None          │ True         │ False        │ None         │
    ├─────────────┼───────────────┼──────────────┼──────────────┼──────────────┤
    │ None        │ sub3 inside   │ None         │ None         │ False        │
    ├─────────────┼───────────────┼──────────────┼──────────────┼──────────────┤
    │ sub2 here   │ no match      │ False        │ True         │ False        │
    ╰─────────────┴───────────────┴──────────────┴──────────────┴──────────────╯
    """

    contains_dict: Dict[str, List[str]]

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "Contains":
        """Fit the transformer (no-op, but required for sklearn compatibility).

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame.
        y : Optional[pl.Series], default=None
            Target variable. Not used, present here for compatibility.

        Returns
        -------
        Contains
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
            pl.col(col).str.contains(substring).alias(f"{col}__contains_{substring}")
            for col, substrings in self.contains_dict.items()
            for substring in substrings
        ]
        return X.with_columns(transformations)
