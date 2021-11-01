# License: Apache-2.0
import warnings
from typing import Dict, List, Union

import databricks.koalas as ks
import numpy as np
import pandas as pd

from ..util import util
from ._base_encoder import _BaseEncoder


class OrdinalEncoder(_BaseEncoder):
    """Encode the categorical columns as integer columns.

    Parameters
    ----------
    dtype : type, default to np.float64.
        Numerical datatype of the output data.
    add_other_columns: bool, default to True.
        If True, add the columns 'OTHERS' and 'MISSING'
        to the mapping even if the categories are not
        present in the data.

    Examples
    ---------

    * fit & transform with `pandas`

    >>> import pandas as pd
    >>> from gators.encoders import OrdinalEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OrdinalEncoder()
    >>> obj.fit_transform(X)
         A    B
    0  1.0  1.0
    1  1.0  0.0
    2  0.0  0.0

    * fit & transform with `koalas`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import OrdinalEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OrdinalEncoder()
    >>> obj.fit_transform(X)
         A    B
    0  1.0  1.0
    1  1.0  0.0
    2  0.0  0.0

    * fit with `pandas` & transform with `NumPy`

    >>> import pandas as pd
    >>> from gators.encoders import OrdinalEncoder
    >>> X = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OrdinalEncoder()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 1.],
           [1., 0.],
           [0., 0.]])

    * fit with `koalas` & transform with `NumPy`

    >>> import databricks.koalas as ks
    >>> from gators.encoders import OrdinalEncoder
    >>> X = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': ['c', 'd', 'd']})
    >>> obj = OrdinalEncoder()
    >>> _ = obj.fit(X)
    >>> obj.transform_numpy(X.to_numpy())
    array([[1., 1.],
           [1., 0.],
           [0., 0.]])
    """

    def __init__(self, dtype: type = np.float64, add_other_columns: bool = True):
        _BaseEncoder.__init__(self, dtype=dtype)
        if not isinstance(add_other_columns, bool):
            raise TypeError("`add_other_columns` shouldbe a bool.")
        self.add_other_columns = add_other_columns

    def fit(
        self,
        X: Union[pd.DataFrame, ks.DataFrame],
        y: Union[pd.Series, ks.Series] = None,
    ) -> "OrdinalEncoder":
        """Fit the transformer on the dataframe `X`.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame].
            Input dataframe.
        y : Union[pd.Series, ks.Series], default to None.
            Labels.

        Returns
        -------
        OrdinalEncoder: Instance of itself.
        """
        self.check_dataframe(X)
        self.columns = util.get_datatype_columns(X, object)
        self.check_nans(X, self.columns)
        if not self.columns:
            warnings.warn(
                f"""`X` does not contain object columns:
                `{self.__class__.__name__}` is not needed"""
            )
            return self
        self.mapping = self.generate_mapping(X, self.columns, self.add_other_columns)
        self.num_categories_vec = np.array([len(m) for m in self.mapping.values()])
        columns, self.values_vec, self.encoded_values_vec = self.decompose_mapping(
            mapping=self.mapping,
        )
        self.idx_columns = util.get_idx_columns(
            columns=X.columns, selected_columns=columns
        )
        return self

    @staticmethod
    def generate_mapping(
        X: Union[pd.DataFrame, ks.DataFrame],
        columns: List[str],
        add_other_columns: bool,
    ) -> Dict[str, Dict[str, float]]:
        """Generate the mapping to perform the encoding.

        Parameters
        ----------
        X : Union[pd.DataFrame, ks.DataFrame]
            Input dataframe.
        self.columns : List[str]
            List of  columns.
        add_other_columns: bool
            If True, add the columns 'OTHERS' and 'MISSING'
            to the mapping even if the categories are not
            present in the data.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping.
        """
        mapping = {}
        for c in columns:
            categories = X[c].value_counts().to_dict()
            n_categories = len(categories)
            category_names = list(categories.keys())
            category_names = sorted(category_names)
            category_mapping = dict(
                zip(category_names, np.arange(n_categories - 1, -1, -1).astype(str))
            )
            if add_other_columns and "MISSING" not in category_mapping:
                category_mapping["MISSING"] = str(len(category_mapping))
            if add_other_columns and "OTHERS" not in category_mapping:
                category_mapping["OTHERS"] = str(len(category_mapping))
            mapping[c] = category_mapping

        return mapping
