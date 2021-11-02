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


def test_check_dataframe_contains_numerics():
    Transformer.check_dataframe_contains_numerics(
        dd.from_pandas(pd.DataFrame({"A": [1], "B": ["b"]}), npartitions=1)
    )


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
    with pytest.raises(ValueError):
        Class().check_array_is_numerics(np.array(["a"]))
    with pytest.raises(ValueError):
        Transformer.check_dataframe_is_numerics(
            dd.from_pandas(pd.DataFrame({"A": [1], "x": ["x"]})), npartitions=1
        )
    with pytest.raises(ValueError):
        Transformer.check_binary_target(
            X, dd.from_pandas(pd.Series([1, 2, 3], name="TARGET"), npartitions=1)
        )
    with pytest.raises(ValueError):
        Transformer.check_multiclass_target(
            dd.from_pandas(pd.Series([1.1, 2.2, 3.3], name="TARGET"), npartitions=1)
        )
    with pytest.raises(ValueError):
        Transformer.check_regression_target(
            dd.from_pandas(pd.Series([1, 0, 0], name="TARGET"), npartitions=1)
        )
    with pytest.raises(ValueError):
        Class().check_nans(
            dd.from_pandas(pd.DataFrame({"A": [np.nan]}), npartitions=1), columns=["A"]
        )
    with pytest.raises(ValueError):
        Class().check_dataframe_with_objects(
            dd.from_pandas(pd.DataFrame({"A": [1.1], "B": [0]}), npartitions=1)
        )
    with pytest.raises(ValueError):
        Class().check_dataframe_contains_numerics(
            dd.from_pandas(pd.DataFrame({"A": ["a"], "B": ["b"]}), npartitions=1)
        )
    with pytest.raises(ValueError):
        Class().check_datatype(object, [np.float64])
    with pytest.raises(ValueError):
        Class().check_array_is_numerics(np.array([["a"], ["b"]]))
