# License: Apache-2.0
import warnings
from typing import Dict

import numpy as np

from ..util import util
from ._base_encoder import _BaseEncoder
from ..util.iv import compute_iv

from gators import DataFrame, Series


class WOEEncoder(_BaseEncoder):
    """Encode all categorical variables using the weight of evidence technique.

    Parameters
    ----------
    regularization : float, default 0.5.
        Insure that the weights of evidence are finite.

    inplace : bool, default to True.
        If True, replace in-place the categorical values by numerical ones.
        If False, keep the categorical columns and create new encoded columns.


    Examples
    --------

    Imports and initialization:

    >>> from gators.encoders import WOEEncoder
    >>> obj = WOEEncoder()

    The `fit`, `transform`, and `fit_transform` methods accept:

    * `dask` dataframes:

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> X = dd.from_pandas(pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']}), npartitions=1)
    >>> y = dd.from_pandas(pd.Series([1, 1, 0], name='TARGET'), npartitions=1)

    * `koalas` dataframes:

    >>> import databricks.koalas as ks
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ks.Series([1, 1, 0], name='TARGET')

    * and `pandas` dataframes:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')

    The result is a transformed dataframe belonging to the same dataframe library.

    >>> obj.fit_transform(X, y)
              A         B
    0  1.203973  0.693147
    1  1.203973 -0.405465
    2 -1.504077 -0.405465

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[ 1.2039728 ,  0.69314718],
           [ 1.2039728 , -0.40546511],
           [-1.5040774 , -0.40546511]])
    """

    def __init__(
        self,
        regularization: float = 0.5,
        inplace: bool = True,
    ):
        if not isinstance(regularization, (int, float)):
            raise TypeError("""`regularization` should be a float.""")
        if regularization < 0:
            raise ValueError("""`regularization` should be a positive float.""")
        self.regularization = regularization
        _BaseEncoder.__init__(self, inplace=inplace)

    def fit(self, X: DataFrame, y: Series) -> "WOEEncoder":
        """Fit the encoder.

        Parameters
        ----------
        X : DataFrame:
            Input dataframe.
        y : Series, default None.
            Target values.

        Returns
        -------
        WOEEncoder:
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_target(X, y)
        self.base_columns = list(X.columns)
        self.columns = util.get_datatype_columns(X, object)
        self.column_names = self.get_column_names(self.inplace, self.columns, "woe")
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(X[self.columns], y)
        self.mapping = {k: self.mapping[k] for k in self.columns}
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        _, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=self.columns
        )
        return self

    def generate_mapping(
        self,
        X: DataFrame,
        y: Series,
    ) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : DataFrame
            Input dataframe.
        y : Series:
             Target values.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        _, stats = compute_iv(X, y, regularization=self.regularization)
        mapping = (
            stats[["woe"]]
            .groupby(level=0)
            .apply(lambda df: df.xs(df.name)["woe"].to_dict())
            .to_dict()
        )
        return mapping
