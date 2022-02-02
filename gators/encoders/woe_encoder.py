# License: Apache-2.0
import warnings
from typing import Dict

import numpy as np

from ..util import util
from ._base_encoder import _BaseEncoder

from gators import DataFrame, Series


class WOEEncoder(_BaseEncoder):
    """Encode all categorical variables using the weight of evidence technique.

    Parameters
    ----------
    regularization : float, default 0.5.
        Insure that the weights of evidence are finite.

    add_missing_categories : bool, default True.
        If True, add the columns 'OTHERS' and 'MISSING'
        to the mapping even if the categories are not
        present in the data.

    dtype : type, default np.float64.
        Numerical datatype of the output data.

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
    0  1.609438  1.098612
    1  1.609438  0.000000
    2 -1.098612  0.000000

    Independly of the dataframe library used to fit the transformer, the `tranform_numpy` method only accepts NumPy arrays
    and returns a transformed NumPy array. Note that this transformer should **only** be used
    when the number of rows is small *e.g.* in real-time environment.

    >>> obj.transform_numpy(X.to_numpy())
    array([[ 1.60943791,  1.09861229],
           [ 1.60943791,  0.        ],
           [-1.09861229,  0.        ]])
    """

    def __init__(
        self,
        regularization: float = 0.5,
        add_missing_categories=True,
        dtype: type = np.float64,
    ):
        if not isinstance(regularization, (int, float)) or regularization < 0:
            raise TypeError("""`min_ratio` should be a positive float.""")
        self.regularization = regularization
        _BaseEncoder.__init__(
            self, add_missing_categories=add_missing_categories, dtype=dtype
        )

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
        self.columns = util.get_datatype_columns(X, object)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(X[self.columns], y)
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
        y_name = y.name
        columns = list(X.columns)
        counts = (
            util.get_function(X)
            .melt(util.get_function(X).join(X, y.to_frame()), id_vars=y_name)
            .groupby(["variable", "value"])
            .agg(["sum", "count"])[y_name]
        )
        counts = util.get_function(X).to_pandas(counts)
        counts.columns = ["1", "count"]
        counts["0"] = (counts["count"] - counts["1"] + self.regularization) / counts[
            "count"
        ]
        counts["1"] = (counts["1"] + self.regularization) / counts["count"]
        woe = np.log(counts["1"] / counts["0"])
        mapping = {c: woe[c].to_dict() for c in columns}

        return self.clean_mapping(mapping, self.add_missing_categories)
