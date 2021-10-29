# License: Apache-2.0
from typing import List, Tuple, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..util import util
from ._base_discretizer import _BaseDiscretizer

EPSILON = 1e-10


class Discretizer(_BaseDiscretizer):
    """Discretize the columns using equal distance splits.

    The discretization can be done inplace or by adding the discretized
    columns to the existing data.

    Parameters
    ----------
    n_bins : int
        Number of bins to use.
    inplace : bool, default False
        If False, return the dataframe with the new discretized columns
        with the names "`column_name`__bin"). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    ---------
    * fit & transform with `pandas`

        - inplace discretization

            >>> import pandas as pd
            >>> from gators.binning import Discretizer
            >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
            >>> obj = Discretizer(n_bins=3, inplace=True)
            >>> obj.fit_transform(X)
                 A    B
            0  0.0  0.0
            1  1.0  1.0
            2  2.0  2.0

        - add discretization

            >>> import pandas as pd
            >>> from gators.binning import Discretizer
            >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
            >>> obj = Discretizer(n_bins=3,  inplace=False)
            >>> obj.fit_transform(X)
               A  B A__bin B__bin
            0 -1  1    0.0    0.0
            1  0  2    1.0    1.0
            2  1  3    2.0    2.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.binning import Discretizer
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = Discretizer(n_bins=3)
    >>> obj.fit_transform(X)
       A  B A__bin B__bin
    0 -1  1    0.0    0.0
    1  0  2    1.0    1.0
    2  1  3    2.0    2.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.binning import Discretizer
    >>> X = pd.DataFrame({'A': [-1., 0., 1.], 'B': [1., 2., 3.]})
    >>> obj = Discretizer(n_bins=3)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.0, 1.0, '0.0', '0.0'],
           [0.0, 2.0, '1.0', '1.0'],
           [1.0, 3.0, '2.0', '2.0']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.binning import Discretizer
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = Discretizer(n_bins=3)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 1, '0.0', '0.0'],
           [0, 2, '1.0', '1.0'],
           [1, 3, '2.0', '2.0']], dtype=object)

    See Also
    --------
    gators.binning.CustomDiscretizer
        Discretize using the splits given by the user.
    gators.binning.QuantileDiscretizer
        Discretize using splits based on quantiles.

    """

    def __init__(self, n_bins: int, inplace=False):
        _BaseDiscretizer.__init__(self, n_bins=n_bins, inplace=inplace)

    @staticmethod
    def compute_bins(
        X: Union[pd.DataFrame, ks.DataFrame], n_bins: int
    ) -> Tuple[List[List[float]], np.ndarray]:
        """Compute the bins list and the bins array.
        The bin list is used for dataframes and
        the bins array is used for arrays.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        n_bins : int
            Number of bins to use.

        Returns
        -------
        bins : List[List[float]]
            Bin splits definition.
            The dictionary keys are the column names to discretize,
            its values are the split arrays.
        bins_np : np.ndarray
            Bin splits definition for NumPy.
        """
        n_cols = X.shape[1]
        X_dtype = X.dtypes.to_numpy()[0]
        if isinstance(X, pd.DataFrame):
            deltas = X.max() - X.min()
            bins_np = np.empty((n_bins + 1, n_cols))
            bins_np[0, :] = util.get_bounds(X_dtype)[0]
            bins_np[-1, :] = util.get_bounds(X_dtype)[1]
            for i in range(1, n_bins):
                bins_np[i, :] = X.min() + i * deltas / n_bins

            bins = pd.DataFrame(bins_np, columns=X.columns).to_dict(orient="list")
            return bins, bins_np
        x_min = X.min().to_pandas()
        x_max = X.max().to_pandas()
        deltas = x_max - x_min
        bins_np = np.empty((n_bins + 1, n_cols))
        bins_np[0, :] = util.get_bounds(X_dtype)[0]
        bins_np[-1, :] = util.get_bounds(X_dtype)[1]
        for i in range(1, n_bins):
            bins_np[i, :] = x_min + i * deltas / n_bins
        bins = (bins_np.T + EPSILON).tolist()
        return bins, bins_np
