# License: Apache-2.0
import warnings
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..util import util
from ._base_encoder import _BaseEncoder


def clean_mapping(
    mapping: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, List[float]]]:
    mapping = {
        col: {k: v for k, v in mapping[col].items() if v == v} for col in mapping.keys()
    }
    for m in mapping.values():
        if "OTHERS" not in m:
            m["OTHERS"] = 0.0
        if "MISSING" not in m:
            m["MISSING"] = 0.0
    return mapping


class WOEEncoder(_BaseEncoder):
    """Encode all categorical variable using the weight of evidence technique.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.

    Examples
    --------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.encoders import WOEEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')
    >>> obj = WOEEncoder()
    >>> obj.fit_transform(X, y)
         A         B
    0  0.0  0.000000
    1  0.0 -0.693147
    2  0.0 -0.693147

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import WOEEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ks.Series([1, 1, 0], name='TARGET')
    >>> obj = WOEEncoder()
    >>> obj.fit_transform(X, y)
         A         B
    0  0.0  0.000000
    1  0.0 -0.693147
    2  0.0 -0.693147

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.encoders import WOEEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = pd.Series([1, 1, 0], name='TARGET')
    >>> obj = WOEEncoder()
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.        ,  0.        ],
           [ 0.        , -0.69314718],
           [ 0.        , -0.69314718]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import WOEEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> y = ks.Series([1, 1, 0], name='TARGET')
    >>> obj = WOEEncoder()
    >>> _ = obj.fit(X, y)
    >>> obj.transform_numpy(X.to_numpy())
    array([[ 0.        ,  0.        ],
           [ 0.        , -0.69314718],
           [ 0.        , -0.69314718]])
    """

    def __init__(self, dtype: type = np.float64):
        _BaseEncoder.__init__(self, dtype=dtype)

    def fit(
        self, X: Union[pd.DataFrame, ks.DataFrame], y: Union[pd.Series, ks.Series]
    ) -> "WOEEncoder":
        """Fit the encoder.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]:
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        WOEEncoder:
            Instance of itself.
        """
        self.check_dataframe(X)
        self.check_y(X, y)
        self.check_binary_target(y)
        self.columns = util.get_datatype_columns(X, object)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.check_binary_target(y)
        self.check_nans(X, self.columns)
        self.mapping = self.generate_mapping(X[self.columns], y)
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        columns, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=columns
        )
        return self

    @staticmethod
    def generate_mapping(
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series],
    ) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        y : Union[pd.Series, ks.Series]:
             Labels.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        mapping_list = []
        y_name = y.name
        X = X.join(y)
        for col in X.columns:
            if isinstance(X, pd.DataFrame):
                tab = X.groupby([col, y_name])[y_name].count().unstack().fillna(0)
            else:
                tab = (
                    X.groupby([col, y_name])[y_name]
                    .count()
                    .unstack()
                    .to_pandas()
                    .fillna(0)
                )
            tab /= tab.sum()
            tab.columns = [int(c) for c in tab.columns]
            with np.errstate(divide="ignore"):
                woe = pd.Series(np.log(tab[1] / tab[0]))
            woe[(woe == np.inf) | (woe == -np.inf)] = 0.0
            mapping_list.append(pd.Series(woe, name=col))
        mapping = pd.concat(mapping_list, axis=1).to_dict()
        X = X.drop(y_name, axis=1)
        return clean_mapping(mapping)
