# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest

from gators.transformers.transformer_xy import TransformerXY


@pytest.mark.koalas
def test_checks():
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe([])
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe(ks.DataFrame({"A": [1], 0: ["x"]}))
    with pytest.raises(TypeError):
        TransformerXY.check_target(ks.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        TransformerXY.check_target(ks.DataFrame({"A": [1], 0: ["x"]}), ks.Series([1]))
    with pytest.raises(TypeError):
        TransformerXY.check_target(ks.DataFrame({"A": [1], "B": [2]}), pd.Series([0]))
