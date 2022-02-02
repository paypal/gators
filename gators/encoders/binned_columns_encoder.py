# License: Apache-2.
import numpy as np

from typing import Dict, List

import numpy as np

from encoder import binned_columns_encoder
from encoder import binned_columns_encoder_inplace

from ..util import util
from ._base_encoder import _BaseEncoder

from gators import DataFrame, Series


class BinnedColumnsEncoder(_BaseEncoder):
    """Encode the categorical variables after running a **Gators** Binning transformer.

    Replace the bins "_X" by *X*, where *X* is an integer.

    Parameters
    ----------
    dtype : type, default np.float64.
        Numerical datatype of the output data.

    Examples
    --------

    Imports and initialization:

    >>> from gators.encoders import BinnedColumnsEncoder
    >>> obj = BinnedColumnsEncoder(columns=['A', 'B'], inplace=False)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['_0', '_0', '_1'], 'B': ['_1', '_0', '_0']}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['_0', '_0', '_1'], 'B': ['_1', '_0', '_0']})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['_0', '_0', '_1'], 'B': ['_1', '_0', '_0']})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X.copy())
        A   B  A__ordinal  B__ordinal
    0  _0  _1         0.0         1.0
    1  _0  _0         0.0         0.0
    2  _1  _0         1.0         0.0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([['_0', '_1', 0.0, 1.0],
           ['_0', '_0', 0.0, 0.0],
           ['_1', '_0', 1.0, 0.0]], dtype=object)
    """

    def __init__(self, columns: List[str], inplace: bool = True):
        if not isinstance(columns, (list, np.ndarray)):
            raise TypeError("`columns` should be a list.")
        if not isinstance(inplace, bool):
            raise TypeError("`inplace` should be a bool.")
        self.columns = columns
        self.inplace = inplace

    def fit(self, X: DataFrame, y: Series = None) -> "BinnedColumnsEncoder":
        """Fit the encoder.

        Parameters
        ----------
        X : DataFrame:
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        BinnedColumnsEncoder:
            Instance of itself.
        """
        self.check_dataframe(X)
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
        )
        return self

    def transform(self, X: DataFrame, y: Series = None) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None
             Target values.

        Returns
        -------
        X : DataFrame
            Transformed dataframe.
        """
        self.check_dataframe(X)
        if not self.columns:
            return X
        if self.inplace:
            for c in self.columns:
                X[c] = X[c].str.slice(start=1).astype(np.float64)
            return X
        for c in self.columns:
            X[f"{c}__ordinal"] = X[c].str.slice(start=1).astype(np.float64)
        self.columns_ = list(X.columns)
        return X

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        self.check_array(X)
        if self.idx_columns.size == 0:
            return X
        if self.inplace:
            return binned_columns_encoder_inplace(X, self.idx_columns)
        return binned_columns_encoder(X, self.idx_columns)
