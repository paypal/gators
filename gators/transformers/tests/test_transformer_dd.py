# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from gators.transformers.transformer import Transformer


class Class(Transformer):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pass

    def transform_numpy(self, X):
        pass


def test_checks():
    X = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(TypeError):
        Transformer.check_dataframe([])
    with pytest.raises(TypeError):
        Transformer.check_dataframe(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1)
        )
    with pytest.raises(TypeError):
        Transformer.check_target(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1), []
        )
    with pytest.raises(TypeError):
        Transformer.check_target(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1),
            dd.from_pandas(pd.Series([1]), npartitions=1),
        )
    with pytest.raises(ValueError):
        Transformer.check_target(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1),
            dd.from_pandas(pd.Series([1, 2], name="Y"), npartitions=1),
        )
    with pytest.raises(TypeError):
        Transformer.check_array([])
    with pytest.raises(TypeError):
        Transformer.check_target(X, [])
