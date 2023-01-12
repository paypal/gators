# License: Apache-2.0
import warnings
from typing import Dict

import numpy as np

from ..util import util
from ._base_encoder import _BaseEncoder

from gators import DataFrame, Series


class OrdinalEncoder(_BaseEncoder):
    """Encode the categorical columns as integer columns.

    Parameters
    ----------
    inplace : bool, default to True.
        If True, replace in-place the categorical values by numerical ones.
        If False, keep the categorical columns and create new encoded columns.


    Examples
    ---------

    Imports and initialization:

    >>> from gators.encoders import OrdinalEncoder
    >>> obj = OrdinalEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B
    0  1.0  0.0
    1  1.0  1.0
    2  0.0  1.0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 0.],
           [1., 1.],
           [0., 1.]])
    """

    def __init__(self, inplace=True):
        _BaseEncoder.__init__(self, inplace=inplace)

    def fit(self, X: DataFrame, y: Series = None) -> "OrdinalEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        OrdinalEncoder: Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.columns = util.get_datatype_columns(X, object)
        self.column_names = self.get_column_names(self.inplace, self.columns, "ordinal")
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(X[self.columns])
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        columns, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping,
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=columns
        )
        return self

    def generate_mapping(self, X: DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        mapping = {}
        for c in X.columns:
            mapping[c] = (
                util.get_function(X)
                .to_pandas(X[c].value_counts(ascending=True))
                .to_dict()
            )
            mapping[c] = {k: float(i) for i, (k, v) in enumerate(mapping[c].items())}
        return mapping
