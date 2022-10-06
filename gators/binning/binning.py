# License: Apache-2.0
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..util import util
from ._base_binning import _BaseBinning

EPSILON = 1e-10

from gators import DataFrame, Series


class Binning(_BaseBinning):
    """Bin the columns using equal distance splits.

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
    Imports and initialization:

    >>> from gators.binning import Binning

    The binning can be done inplace by modifying the existing columns

    >>> obj = Binning(n_bins=3, inplace=True)

    or by adding new binned columns

    >>> obj = Binning(n_bins=3, inplace=False)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 1, 0]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [3, 1, 0]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 1, 0]})

    The result is a transformed dataframe belonging to the same dataframe library.

    * with `inplace=True`

    >>> obj = Binning(n_bins=3, inplace=True)
    >>> obj.fit_transform(X)
                   A            B
    0  (-inf, -0.33)   [2.0, inf)
    1  [-0.33, 0.33)   [1.0, 2.0)
    2    [0.33, inf)  (-inf, 1.0)

    * with `inplace=False`

    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 1, 0]})
    >>> obj = Binning(n_bins=3, inplace=False)
    >>> obj.fit_transform(X)
       A  B         A__bin       B__bin
    0 -1  3  (-inf, -0.33)   [2.0, inf)
    1  0  1  [-0.33, 0.33)   [1.0, 2.0)
    2  1  0    [0.33, inf)  (-inf, 1.0)

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 1, 0]})
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 3, '(-inf, -0.33)', '[2.0, inf)'],
           [0, 1, '[-0.33, 0.33)', '[1.0, 2.0)'],
           [1, 0, '[0.33, inf)', '(-inf, 1.0)']], dtype=object)

    See Also
    --------
    gators.binning.CustomBinning
        Bin using the splits given by the user.
    gators.binning.QuantileBinning
        Bin using splits based on quantiles.
    gators.binning.TreeBinning
        Bin using splits based on decision trees.

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
        n_cols = X.shape[1]
        bins_np = np.empty((self.n_bins + 1, n_cols))
        bins_np[0, :] = -np.inf
        bins_np[-1, :] = np.inf
        x_min = util.get_function(X).to_pandas(X.min())
        x_max = util.get_function(X).to_pandas(X.max())
        deltas = x_max - x_min
        for i in range(1, self.n_bins):
            bins_np[i, :] = x_min + i * deltas / self.n_bins
        bins_dict = pd.DataFrame(bins_np, columns=X.columns).to_dict(orient="list")
        bins_dict = {k: np.unique(v) for k, v in bins_dict.items()}
        pretty_bins_dict = {
            k: [util.prettify_number(x, precision=2) for x in v]
            for k, v in bins_dict.items()
        }
        return bins_dict, pretty_bins_dict, bins_np
