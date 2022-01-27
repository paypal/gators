# License: Apache-2.0
import databricks.koalas as ks
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


@pytest.mark.koalas
def test_check_dataframe_contains_numerics():
    Transformer.check_dataframe_contains_numerics(ks.DataFrame({"A": [1], "B": ["b"]}))


@pytest.mark.koalas
def test_checks():
    X = ks.DataFrame()
    with pytest.raises(TypeError):
        Transformer.check_dataframe([])
    with pytest.raises(TypeError):
        Transformer.check_dataframe(ks.DataFrame({"A": [1], 0: ["x"]}))
    with pytest.raises(TypeError):
        Transformer.check_target(ks.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        Transformer.check_target(ks.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        Transformer.check_target(ks.DataFrame({"A": [1], 0: ["x"]}), ks.Series([1]))
    with pytest.raises(ValueError):
        Transformer.check_target(
            ks.DataFrame({"A": [1], 0: ["x"]}), ks.Series([1, 2], name="Y")
        )
    with pytest.raises(TypeError):
        Transformer.check_array([])
    with pytest.raises(TypeError):
        Transformer.check_target(X, [])
    with pytest.raises(ValueError):
        Class().check_array_is_numerics(np.array(["a"]))
    with pytest.raises(ValueError):
        Transformer.check_dataframe_is_numerics(ks.DataFrame({"A": [1], "x": ["x"]}))
    with pytest.raises(ValueError):
        Transformer.check_binary_target(X, ks.Series([1, 2, 3], name="TARGET"))
    with pytest.raises(ValueError):
        Transformer.check_multiclass_target(ks.Series([1.1, 2.2, 3.3], name="TARGET"))
    with pytest.raises(ValueError):
        Transformer.check_regression_target(ks.Series([1, 0, 0], name="TARGET"))
    with pytest.raises(ValueError):
        Class().check_nans(ks.DataFrame({"A": [np.nan]}), columns=["A"])
    with pytest.raises(ValueError):
        Class().check_dataframe_with_objects(ks.DataFrame({"A": [1.1], "B": [0]}))
    with pytest.raises(ValueError):
        Class().check_dataframe_contains_numerics(
            ks.DataFrame({"A": ["a"], "B": ["b"]})
        )
    with pytest.raises(ValueError):
        Class().check_datatype(object, [np.float64])
    with pytest.raises(ValueError):
        Class().check_array_is_numerics(np.array([["a"], ["b"]]))
