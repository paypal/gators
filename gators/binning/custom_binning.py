# License: Apache-2.0
from typing import Dict, List

import pandas as pd
import numpy as np

from ..util import util
from ._base_binning import _BaseBinning

EPSILON = 1e-10


from gators import DataFrame, Series


class CustomBinning(_BaseBinning):
    """Bin the columns using the splits given by the user.

    The binning can be done inplace or by adding the binned
    columns to the existing data.

    Parameters
    ----------
    bins_dict : Dict[str, List[float]]
        Bin splits definition. The dictionary keys are the column names to
        bin, its values are the split arrays.
    inplace : bool, default False
        If False, return the dataframe with the new binned columns
        with the names "column_name__bin"). Otherwise, return
        the dataframe with the existing binned columns.

    Examples
    --------
    >>> import numpy as np
    >>> from gators.binning import Binning

    >>> bins_dict = {'A':[-np.inf, 1, np.inf], 'B':[-np.inf, 2, np.inf]}

    The binning can be done inplace by modifying the existing columns:

    >>> obj = CustomBinning(bins_dict=bins_dict, inplace=True)

    or by adding new binned columns:

    >>> obj = CustomBinning(bins_dict=bins_dict, inplace=True)

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

    >>> obj = CustomBinning(bins_dict=bins_dict, inplace=True)
    >>> obj.fit_transform(X)
                 A            B
    0  (-inf, 1.0)   [2.0, inf)
    1  (-inf, 1.0)   [2.0, inf)
    2   [1.0, inf)  (-inf, 2.0)

    * with `inplace=False`

    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]})
    >>> obj = CustomBinning(bins_dict=bins_dict, inplace=False)
    >>> obj.fit_transform(X)
       A  B       A__bin       B__bin
    0 -1  3  (-inf, 1.0)   [2.0, inf)
    1  0  2  (-inf, 1.0)   [2.0, inf)
    2  1  1   [1.0, inf)  (-inf, 2.0)

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> X = pd.DataFrame({'A': [-1, 0, 1], 'B': [3, 2, 1]})
    >>> obj.transform_numpy(X.to_numpy())
    array([[-1, 3, '(-inf, 1.0)', '[2.0, inf)'],
           [0, 2, '(-inf, 1.0)', '[2.0, inf)'],
           [1, 1, '[1.0, inf)', '(-inf, 2.0)']], dtype=object)


    See Also
    --------
    gators.binning.Binning
        Bin using equal splits.
    gators.binning.QuantileBinning
        Bin using splits based on quantiles.
    gators.binning.TreeBinning
        Bin using tree-based splits.
    """

    def __init__(self, bins_dict: Dict[str, List[float]], inplace=False):
        if not isinstance(bins_dict, dict):
            raise TypeError("`bins_dict` should be a dict.")
        _BaseBinning.__init__(self, n_bins=1, inplace=inplace)
        self.bins_dict = {key: np.array(val) for key, val in bins_dict.items()}

    def fit(self, X: DataFrame, y: Series = None) -> "CustomBinning":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : 'CustomBinning'
            Instance of itself.
        """
        self.check_dataframe(X)
        self.base_columns = list(X.columns)
        self.columns = list(self.bins_dict.keys())
        self.column_names = [f"{c}__bin" for c in self.columns]
        self.idx_columns = util.get_idx_columns(X.columns, self.columns)
        n_cols = self.idx_columns.size
        if n_cols == 0:
            return self
        self.pretty_bins_dict = {
            k: [util.prettify_number(x, precision=2) for x in v]
            for k, v in self.bins_dict.items()
        }
        self.labels, self.labels_np = self.get_labels(self.pretty_bins_dict)
        max_bins = max([len(v) for v in self.bins_dict.values()])
        self.bins_np = np.inf * np.ones((max_bins, n_cols))
        for i, b in enumerate(self.bins_dict.values()):
            self.bins_np[: len(b), i] = b
        return self
