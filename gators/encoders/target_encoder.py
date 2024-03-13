# License: Apache-2.
import warnings
from typing import Dict

import numpy as np

from ..util import util
from ._base_encoder import _BaseEncoder

from gators import DataFrame, Series


class TargetEncoder(_BaseEncoder):
    """Encode the categorical variables using the target encoding technique.

    Parameters
    ----------
    inplace : bool, default to True.
        If True, replace in-place the categorical values by numerical ones.
        If False, keep the categorical columns and create new encoded columns.


    Examples
    --------

    Imports and initialization:

    >>> from gators.encoders import TargetEncoder
    >>> obj = TargetEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([1, 1, 0], name='TARGET'), npartitions=1)

    * `koalas` dataframes:

    >>> import pyspark.pandas as ps
    >>> X = ps.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ps.Series([1, 1, 0], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X, y)
         A    B
    0  1.0  1.0
    1  1.0  0.5
    2  0.0  0.5

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[1. , 1. ],
           [1. , 0.5],
           [0. , 0.5]])
    """

    def __init__(self, columns=None, inplace=True):
        _BaseEncoder.__init__(self, columns=None, inplace=inplace)
        self.suffix = "target"

    def generate_mapping(self, X: DataFrame, y: Series) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series:
             Target values.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        columns = X.columns
        means = (
            util.get_function(X)
            .melt(util.get_function(X).join(X, y.to_frame()), id_vars=y.name)
            .groupby(["variable", "value"])
            .mean()[y.name]
        )
        means = util.get_function(X).to_pandas(means)
        return {c: means[c].to_dict() for c in columns}
