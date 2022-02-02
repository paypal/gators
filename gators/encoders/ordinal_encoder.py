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
    dtype : type, default np.float64.
        Numerical datatype of the output data.

    add_missing_categories : bool, default True.
        If True, add the columns 'OTHERS' and 'MISSING'
        to the mapping even if the categories are not
        present in the data.

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

    def __init__(self, dtype: type = np.float64, add_missing_categories: bool = True):
        _BaseEncoder.__init__(
            self, dtype=dtype, add_missing_categories=add_missing_categories
        )

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
        self.columns = util.get_datatype_columns(X, object)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(
            X[self.columns], self.add_missing_categories
        )
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        columns, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping,
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=columns
        )
        return self

    def generate_mapping(
        self, X: DataFrame, add_missing_categories: bool
    ) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        add_missing_categories: bool
            If True, add the columns 'OTHERS' and 'MISSING'
            to the mapping even if the categories are not
            present in the data.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        size = (
            util.get_function(X)
            .to_pandas(
                util.get_function(X).melt(X).groupby(["variable", "value"]).size()
            )
            .sort_values()
        )
        mapping = {}
        for c in X.columns:
            mapping[c] = dict(zip(size.loc[c].index, range(len(size.loc[c].index))))
            if add_missing_categories and "MISSING" not in mapping[c]:
                mapping[c]["MISSING"] = -1.0
            if add_missing_categories and "OTHERS" not in mapping[c]:
                mapping[c]["OTHERS"] = -1.0
        return mapping
