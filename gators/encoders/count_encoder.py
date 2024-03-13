# License: Apache-2.0
import warnings
from typing import Dict

from ..util import util
from ._base_encoder import _BaseEncoder

from gators import DataFrame, Series


class CountEncoder(_BaseEncoder):
    """Encode the categorical columns as integer columns based on the category count.

    Parameters
    ----------
    inplace : bool, default to True.
        If True, replace in-place the categorical values by numerical ones.
        If False, keep the categorical columns and create new encoded columns.

    Examples
    ---------

    Imports and initialization:

    >>> from gators.encoders import CountEncoder
    >>> obj = CountEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)

    * `koalas` dataframes:

    >>> import pyspark.pandas as ps
    >>> X = ps.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         A    B
    0  2.0  1.0
    1  2.0  2.0
    2  1.0  2.0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[2., 1.],
           [2., 2.],
           [1., 2.]])
    """

    def __init__(self, columns=None, inplace=True):
        _BaseEncoder.__init__(self, columns=columns, inplace=inplace)
        self.suffix = "count"

    def generate_mapping(
        self, X: DataFrame, y: Series = None
    ) -> Dict[str, Dict[str, float]]:
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
        return {
            c: (
                util.get_function(X)
                .to_pandas(X[c].value_counts())
                .astype(float)
                .to_dict()
            )
            for c in X.columns
        }
