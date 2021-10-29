# License: Apache-2.0
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..util import util
from ._base_discretizer import _BaseDiscretizer

EPSILON = 1e-10


class CustomDiscretizer(_BaseDiscretizer):
    """Discretize the columns using the splits given by the user.

    The discretization can be done inplace or by adding the discretized
    columns to the existing data.

    Parameters
    ----------
    bins : Dict[str, List[float]]
        Bin splits definition. The dictionary keys are the column names to
        discretize, its values are the split arrays.
    inplace : bool, default False
        If False, return the dataframe with the new discretized columns
        with the names "`column_name`__bin"). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    --------
    * fit & transform with `pandas`

        - inplace discretization
            >>> import pandas as pd
            >>> import numpy as np
            >>> from gators.binning import CustomDiscretizer
            >>> bins = {'A':[-np.inf, 0, np.inf], 'B':[-np.inf, 1, np.inf]}
            >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
            >>> obj = CustomDiscretizer(bins=bins, inplace=True)
            >>> obj.fit_transform(X)
                 A    B
            0  0.0  0.0
            1  0.0  1.0
            2  1.0  1.0

        - add discretization

            >>> import pandas as pd
            >>> import numpy as np
            >>> from gators.binning import CustomDiscretizer
            >>> bins = {'A':[-np.inf, 0, np.inf], 'B':[-np.inf, 1, np.inf]}
            >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
            >>> obj = CustomDiscretizer(bins=bins, inplace=False)
            >>> obj.fit_transform(X)
               A  B A__bin B__bin
            0 -1  1    0.0    0.0
            1  0  2    0.0    1.0
            2  1  3    1.0    1.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.binning import CustomDiscretizer
    >>> bins = {'A':[-np.inf, 0, np.inf], 'B':[-np.inf, 1, np.inf]}
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = CustomDiscretizer(bins=bins)
    >>> obj.fit_transform(X)
       A  B A__bin B__bin
    0 -1  1    0.0    0.0
    1  0  2    0.0    1.0
    2  1  3    1.0    1.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gators.binning import CustomDiscretizer
    >>> bins = {'A':[-np.inf, 0, np.inf], 'B':[-np.inf, 1, np.inf]}
    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = CustomDiscretizer(bins=bins)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 1, '0.0', '0.0'],
           [0, 2, '0.0', '1.0'],
           [1, 3, '1.0', '1.0']], dtype=object)

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> import numpy as np
    >>> from gators.binning import CustomDiscretizer
    >>> bins = {'A':[-np.inf, 0, np.inf], 'B':[-np.inf, 1, np.inf]}
    >>> X = ks.DataFrame({'A': [-1, 0, 1], 'B': [1, 2, 3]})
    >>> obj = CustomDiscretizer(bins=bins)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 1, '0.0', '0.0'],
           [0, 2, '0.0', '1.0'],
           [1, 3, '1.0', '1.0']], dtype=object)

    See Also
    --------
    gators.binning.Discretizer
        Discretize using equal splits.
    gators.binning.QuantileDiscretizer
        Discretize using splits based on quantiles.

    """

    def __init__(self, bins: Dict[str, List[float]], inplace=False):
        if not isinstance(bins, dict):
            raise TypeError("`bins` should be a dict.")
        _BaseDiscretizer.__init__(self, n_bins=0, inplace=inplace)
        self.bins = {key: np.array(val) for key, val in bins.items()}

    def fit(self, X: Union[pd.DataFrame, ks.DataFrame], y=None) -> "CustomDiscretizer":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        'CustomDiscretizer'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = list(self.bins.keys())
        self.output_columns = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        n_cols = len(self.idx_columns)
        if n_cols == 0:
            return self
        max_bins = max([len(v) for v in self.bins.values()])
        self.labels = np.arange(max_bins - 1)
        self.bins_np = np.inf * np.ones((max_bins, n_cols))
        for i, b in enumerate(self.bins.values()):
            self.bins_np[: len(b), i] = b
        if isinstance(X, ks.DataFrame):
            self.bins = self.bins_np.T.tolist()
            self.bins = [np.unique(b) + EPSILON for b in self.bins]
        return self
