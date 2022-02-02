# License: Apache-2.0
from typing import List, Tuple

import numpy as np

from ..util import util
from ._base_binning import _BaseBinning
from gators import DataFrame, Series


class QuantileBinning(_BaseBinning):
    """Bin the columns using quantile-based splits.

    The binning can be done inplace or by adding the binned
    columns to the existing data.

    Parameters
    ----------
    n_bins : int
        Number of bins to use.
    inplace : bool, default False
        If False, return the dataframe with the new binned columns
        with the names "column_name__bin"). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    ---------
    >>> from gators.binning import QuantileBinning

    The binning can be done inplace by modifying the existing columns

    >>> obj = QuantileBinning(n_bins=3, inplace=True)

    or by adding new binned columns

    >>> obj = QuantileBinning(n_bins=3, inplace=True)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]})

    The result is a transformed dataframe belonging to the same dataframe library.

    * with `inplace=True`

    >>> obj = QuantileBinning(n_bins=3, inplace=True)
    >>> obj.fit_transform(X)
        A   B
    0  _0  _2
    1  _1  _1
    2  _2  _0

    * with `inplace=False`

    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]})
    >>> obj = QuantileBinning(n_bins=3, inplace=False)
    >>> obj.fit_transform(X)
       A  B A__bin B__bin
    0 -1  3     _0     _2
    1  0  2     _1     _1
    2  1  1     _2     _0

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]})
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 3, '_0', '_2'],
           [0, 2, '_1', '_1'],
           [1, 1, '_2', '_0']], dtype=object)

    See Also
    --------
    gators.binning.Binning
        Bin using equal splits.
    gators.binning.CustomBinning
        Bin using the variable quantiles.
    gators.binning.TreeBinning
        Bin using tree-based splits.
    """

    def __init__(self, n_bins: int, inplace=False):
        _BaseBinning.__init__(self, n_bins=n_bins, inplace=inplace)

    def compute_bins(
        self, X: DataFrame, y: Series = None
    ) -> Tuple[List[List[float]], np.ndarray]:
        """Compute the bins list and the bins array.
        The bin list is used for dataframes and
        the bins array is used for arrays.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        n_bins : int
            Number of bins to use.

        Returns
        -------
        bins : List[List[float]]
            Bin splits definition.
            The dictionary keys are the column names to bin,
            its values are the split arrays.
        bins_np : np.ndarray
            Bin splits definition for NumPy.
        """
        q = np.linspace(0.001, 0.999, self.n_bins + 1)[1:-1].tolist()
        X_dtype = X.dtypes.to_numpy()[0]
        bins = X.quantile(q=q)
        bins = util.get_function(bins).to_pandas(bins)

        bins.loc[-np.inf, :] = util.get_bounds(X_dtype)[0]
        infinity = util.get_bounds(X_dtype)[1]
        bins.loc[np.inf, :] = infinity
        bins = bins.sort_index()
        for c in X.columns:
            unique_bins = bins[c].iloc[1:-1].unique()
            n_unique = unique_bins.shape[0]
            bins[c].iloc[1 : 1 + n_unique] = unique_bins
            bins[c].iloc[1 + n_unique :] = infinity

        bins = bins.applymap(
            lambda x: util.prettify_number(x, precision=2, infinity=infinity)
        )
        bins_np = bins.to_numpy()
        bins = bins.to_dict(orient="list")
        bins = {col: np.unique(bins[col]) for col in bins.keys()}
        return bins, bins_np
