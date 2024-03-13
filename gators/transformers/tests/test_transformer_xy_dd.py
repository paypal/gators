# License: Apache-2.0
import dask.dataframe as dd
import pandas as pd
import pytest

from gators.transformers.transformer_xy import TransformerXY


def test_checks():
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe([])
    with pytest.raises(TypeError):
        TransformerXY.check_dataframe(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1)
        )
    with pytest.raises(TypeError):
        TransformerXY.check_target(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1), []
        )
    with pytest.raises(TypeError):
        TransformerXY.check_target(
            dd.from_pandas(pd.DataFrame({"A": [1], 0: ["x"]}), npartitions=1),
            dd.from_pandas(pd.Series([1]), npartitions=1),
        )
    with pytest.raises(TypeError):
        TransformerXY.check_target(pd.DataFrame({"A": [1], "B": [2]}), pd.Series([0]))
