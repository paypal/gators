# License: Apache-2.0
import pyspark.pandas as ps
import pandas as pd
import pytest

from gators.transformers.transformer_xy import TransformerXY


@pytest.mark.pyspark
def test_checks():
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe([])
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe(ps.DataFrame({"A": [1], 0: ["x"]}))
    with pytest.raises(TypeError):
        TransformerXY.check_target(ps.DataFrame({"A": [1], 0: ["x"]}), [])
    with pytest.raises(TypeError):
        TransformerXY.check_target(ps.DataFrame({"A": [1], 0: ["x"]}), ps.Series([1]))
    with pytest.raises(TypeError):
        TransformerXY.check_target(ps.DataFrame({"A": [1], "B": [2]}), pd.Series([0]))
