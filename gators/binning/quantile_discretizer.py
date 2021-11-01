# License: Apache-2.0
from typing import List, Tuple, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..util import util
from ._base_discretizer import _BaseDiscretizer

EPSILON = 1e-10


class QuantileDiscretizer(_BaseDiscretizer):
    """Discretize the columns using quantile-based splits.

    The discretization can be done inplace or by adding the discretized
    columns to the existing data.

    Parameters
    ----------
    n_bins : int
        Number of bins to use.
    inplace : bool, default to False
        If False, return the dataframe with the new discretized columns
        with the names '`column_name`__bin'). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    ---------
    * fit & transform with `pandas`

        - inplace discretization

            >>> import pandas as pd
            >>> from gators.binning import QuantileDiscretizer
            >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
            >>> obj = QuantileDiscretizer(n_bins=3, inplace=True)
            >>> obj.fit_transform(X)
                 A    B
            0  0.0  0.0
            1  1.0  1.0
            2  2.0  2.0

        - add discretization

            >>> import pandas as pd
            >>> from gators.binning import QuantileDiscretizer
            >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
            >>> obj = QuantileDiscretizer(n_bins=3, inplace=False)
            >>> obj.fit_transform(X)
               A  B A__bin B__bin
            0 -1  1    0.0    0.0
            1  0  2    1.0    1.0
            2  1  3    2.0    2.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.binning import QuantileDiscretizer
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = QuantileDiscretizer(n_bins=3)
    >>> obj.fit_transform(X)
       A  B A__bin B__bin
    0 -1  1    0.0    0.0
    1  0  2    1.0    1.0
    2  1  3    2.0    2.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.binning import QuantileDiscretizer
    >>> X = pd.DataFrame({'A': [-1., 0., 1.], 'B': [1., 2., 3.]})
    >>> obj = QuantileDiscretizer(n_bins=3)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1.0, 1.0, '0.0', '0.0'],
           [0.0, 2.0, '1.0', '1.0'],
           [1.0, 3.0, '2.0', '2.0']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.binning import QuantileDiscretizer
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = QuantileDiscretizer(n_bins=3)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 1, '0.0', '0.0'],
           [0, 2, '1.0', '1.0'],
           [1, 3, '2.0', '2.0']], dtype=object)

    See Also
    --------
    gators.binning.Discretizer
        Discretize using equal splits.
    gators.binning.CustomDiscretizer
        Discretize using the variable quantiles.

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
        q = np.linspace(0, 1, n_bins + 1)[1:-1].tolist()
        X_dtype = X.dtypes.to_numpy()[0]

        def f(x):
            return x.quantile(q=q)

        bins = X.apply(f)
        if isinstance(bins, ks.DataFrame):
            bins = bins.to_pandas()
        bins.loc[-np.inf, :] = util.get_bounds(X_dtype)[0]
        bins.loc[np.inf, :] = util.get_bounds(X_dtype)[1]
        bins = bins.sort_index()
        for c in X.columns:
            unique_bins = bins[c].iloc[1:-1].unique()
            n_unique = unique_bins.shape[0]
            bins[c].iloc[1 : 1 + n_unique] = unique_bins
            bins[c].iloc[1 + n_unique :] = util.get_bounds(X_dtype)[1]
        bins_np = bins.to_numpy()
        if isinstance(X, pd.DataFrame):
            return bins.to_dict(orient="list"), bins_np
        else:
            bins = bins_np.T.tolist()
            return [np.unique(b) + EPSILON for b in bins], bins_np
