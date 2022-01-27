# License: Apache-2.0
import pandas as pd
import pytest

from gators.transformers.transformer_xy import TransformerXY


def test_no_transform_method():
    with pytest.raises(TypeError):

        class Class(TransformerXY):
            def fit(self):
                pass

        Class()


def test_object_creation():
    class Class(TransformerXY):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            pass

        def transform_numpy(self, X):
            pass

    Class().fit(0).transform(0)
    assert True


def test_checks():
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe([])
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe(pd.DataFrame({"A": [1], 0: ["x"]}))
    with pytest.raises(TypeError):
        TransformerXY.check_target(pd.DataFrame({"A": [1], 0: ["x"]}), pd.Series([1]))
    with pytest.raises(TypeError):
        TransformerXY.check_target(pd.DataFrame({"A": [1], "B": [2]}), [0])
    with pytest.raises(ValueError):
        TransformerXY.check_target(
            pd.DataFrame({"A": [1], "B": [2]}), pd.Series([1, 2], name="Y")
        )
