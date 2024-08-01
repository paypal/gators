# License: Apache-2.0
import pyspark.pandas as ps
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


@pytest.mark.pyspark
def test_checks():
    X = ps.DataFrame()
    with pytest.raises(TypeError):
        Transformer.check_dataframe([])
    with pytest.raises(TypeError):
        Transformer.check_dataframe(ps.DataFrame({"A": [1], 0: ["x"]}))
    with pytest.raises(TypeError):
        Transformer.check_target(ps.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        Transformer.check_target(ps.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        Transformer.check_target(ps.DataFrame({"A": [1], 0: ["x"]}), ps.Series([1]))
    with pytest.raises(ValueError):
        Transformer.check_target(
            ps.DataFrame({"A": [1], 0: ["x"]}), ps.Series([1, 2], name="Y")
        )
    with pytest.raises(TypeError):
        Transformer.check_array([])
    with pytest.raises(TypeError):
        Transformer.check_target(X, [])
