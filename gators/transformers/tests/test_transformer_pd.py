# License: Apache-2.0
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


def test_no_fit_method():
    with pytest.raises(TypeError):

        class Class(Transformer):
            pass

        Class()


def test_no_transform_method():
    with pytest.raises(TypeError):

        class Class(Transformer):
            def fit(self):
                pass

        Class()


def test_object_creation():
    class Class(Transformer):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            pass

        def transform_numpy(self, X):
            pass

    Class().fit(0).transform(0)
    assert True


def test_object_creation_with_fit_transform():
    class Class(Transformer):
        def fit(self, X, y_none=None):
            return self

        def transform(self, X):
            pass

        def transform_numpy(self, X):
            pass

    Class().fit_transform(0)
    assert True


def test_check_dataframe_contains_numerics():
    Transformer.check_dataframe_contains_numerics(pd.DataFrame({"A": [1], "B": ["b"]}))


def test_checks():
    X = pd.DataFrame()
    with pytest.raises(TypeError):
        Transformer.check_dataframe([])
    with pytest.raises(TypeError):
        Transformer.check_dataframe(pd.DataFrame({"A": [1], 0: ["x"]}))
    with pytest.raises(TypeError):
        Transformer.check_target(pd.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        Transformer.check_target(pd.DataFrame({"A": [1], 0: ["x"]}), pd.Series([1]))
    with pytest.raises(ValueError):
        Transformer.check_target(
            pd.DataFrame({"A": [1], 0: ["x"]}), pd.Series([1, 2], name="Y")
        )
    with pytest.raises(TypeError):
        Transformer.check_array([])
    with pytest.raises(TypeError):
        Transformer.check_target(X, [])
    with pytest.raises(ValueError):
        Class().check_array_is_numerics(np.array(["a"]))
    with pytest.raises(ValueError):
        Transformer.check_dataframe_is_numerics(pd.DataFrame({"A": [1], "x": ["x"]}))
    with pytest.raises(ValueError):
        Transformer.check_binary_target(X, pd.Series([1, 2, 3], name="TARGET"))
    with pytest.raises(ValueError):
        Transformer.check_multiclass_target(pd.Series([1.1, 2.2, 3.3], name="TARGET"))
    with pytest.raises(ValueError):
        Transformer.check_regression_target(pd.Series([1, 0, 0], name="TARGET"))
    with pytest.raises(ValueError):
        Class().check_nans(pd.DataFrame({"A": [np.nan]}), columns=["A"])
    with pytest.raises(ValueError):
        Class().check_dataframe_with_objects(pd.DataFrame({"A": [1.1], "B": [0]}))
    with pytest.raises(ValueError):
        Class().check_dataframe_contains_numerics(
            pd.DataFrame({"A": ["a"], "B": ["b"]})
        )
    with pytest.raises(ValueError):
        Class().check_datatype(object, [np.float64])
    with pytest.raises(ValueError):
        Class().check_array_is_numerics(np.array([["a"], ["b"]]))
