# License: Apache-2.0
from typing import Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..util import util
from ._base_feature_selection import _BaseFeatureSelection


class CorrelationFilter(_BaseFeatureSelection):
    """Remove highly correlated columns.

    Parameters
    ----------
    max_corr : float
        Max correlation value tolerated between two columns.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.feature_selection import CorrelationFilter
    >>> X = pd.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = CorrelationFilter(max_corr=0.9)
    >>> obj.fit_transform(X)
         B     C
    0  1.0  0.00
    1  2.0  0.00
    2  3.0  0.15

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.feature_selection import CorrelationFilter
    >>> X = ks.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = CorrelationFilter(max_corr=0.9)
    >>> obj.fit_transform(X)
         B     C
    0  1.0  0.00
    1  2.0  0.00
    2  3.0  0.15

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.feature_selection import CorrelationFilter
    >>> X = pd.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = CorrelationFilter(max_corr=0.9)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.  , 0.  ],
           [2.  , 0.  ],
           [3.  , 0.15]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.feature_selection import CorrelationFilter
    >>> X = ks.DataFrame(
    ... {'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> obj = CorrelationFilter(max_corr=0.9)
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.  , 0.  ],
           [2.  , 0.  ],
           [3.  , 0.15]])

    """

    def __init__(self, max_corr: float):
        if not isinstance(max_corr, float):
            raise TypeError("`max_corr` should be a float.")
        _BaseFeatureSelection.__init__(self)
        self.max_corr = max_corr

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "CorrelationFilter":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : None
            None.

        Returns
        -------
            CorrelationFilter: Instance of itself.
        """
        self.check_dataframe(X)
        columns = X.columns
        corr = X.corr().abs()
        if not isinstance(X, pd.DataFrame):
            corr = corr.to_pandas()
        stacked_corr = (
            corr.where(np.tril(np.ones(corr.shape), k=-1).astype(np.bool))
            .stack()
            .sort_values(ascending=False)
        )
        stacked_corr = stacked_corr.sort_values(ascending=False)
        mask = stacked_corr >= self.max_corr
        self.columns_to_drop = stacked_corr[mask].sort_index().index.get_level_values(1)
        self.columns_to_drop = list(set(self.columns_to_drop))
        self.selected_columns = [c for c in columns if c not in self.columns_to_drop]
        self.feature_importances_ = pd.Series(
            1.0, index=self.selected_columns, dtype=float
        )
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self
