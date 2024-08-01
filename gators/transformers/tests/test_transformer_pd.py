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
