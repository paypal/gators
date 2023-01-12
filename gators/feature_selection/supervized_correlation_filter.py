# License: Apache-2.0
import pandas as pd

from ..util import util
from ._base_feature_selection import _BaseFeatureSelection

from gators import DataFrame, Series


class SupervizedCorrelationFilter(_BaseFeatureSelection):
    """Remove highly correlated columns.

    Select the features based on the highest feature importance.

    Parameters
    ----------
    feature_importances: Series
        Feature importances.
    max_corr : float
        Max correlation value tolerated between two columns.
    method: str or callable
        Method of correlation:

        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation

    Examples
    ---------
    Imports and initialization:

    >>> from gators.feature_selection import SupervizedCorrelationFilter
    >>> feature_importances = pd.Series({'A': 0.1, 'B': 0.7, 'C':0.9})
    >>> obj = SupervizedCorrelationFilter(max_corr=0.9, feature_importances=feature_importances)

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]}), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X)
         B     C
    0  1.0  0.00
    1  2.0  0.00
    2  3.0  0.15

    >>> X = pd.DataFrame({'A': [0., 0., 0.1], 'B': [1., 2., 3.], 'C': [0., 0., 0.15]})
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1.  , 0.  ],
           [2.  , 0.  ],
           [3.  , 0.15]])
    """

    def __init__(
        self, feature_importances: Series, max_corr: float, method: str = "pearson"
    ):
        if "Series" not in str(type(feature_importances)):
            raise TypeError(
                "`feature_importances` should be a pandas, dask, or koalas Series."
            )
        if not isinstance(max_corr, float):
            raise TypeError("`max_corr` should be a float.")
        _BaseFeatureSelection.__init__(self)
        self.max_corr = max_corr
        self.method = method
        self.feature_importances = util.get_function(feature_importances).to_pandas(
            feature_importances
        )

    def fit(self, X: DataFrame, y: Series = None) -> "SupervizedCorrelationFilter":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : DataFrame.
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        self : "CorrelationFilter"
            Instance of itself.
        """
        self.check_dataframe(X)
        columns = X.columns
        corr = util.get_function(X).to_pandas(X.corr()).abs()
        self.columns_to_drop = []
        for i in range(len(columns)):
            for j in range(i + 1):
                item = corr.iloc[j : (j + 1), (i + 1) : (i + 2)]
                col = item.columns
                row = item.index
                val = item.values
                if val >= self.max_corr:
                    col_value_corr = self.feature_importances[col.values[0]]
                    row_value_corr = self.feature_importances[row.values[0]]
                    if col_value_corr < row_value_corr:
                        self.columns_to_drop.append(col.values[0])
                    else:
                        self.columns_to_drop.append(row.values[0])

        self.columns_to_drop = list(set(self.columns_to_drop))
        self.selected_columns = [c for c in columns if c not in self.columns_to_drop]
        self.feature_importances_ = pd.Series(
            1.0, index=self.selected_columns, dtype=float
        )
        self.idx_selected_columns = util.get_idx_columns(
            X.columns, self.selected_columns
        )
        return self
