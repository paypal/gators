# Licence Apache-2.0
import pytest
import pandas as pd
from gators.feature_generation_dt.ordinal_day_of_month import OrdinalDayOfMonth


def test_init_pd():
    X = pd.DataFrame({"A": [0], "B": [0]})
    obj = OrdinalDayOfMonth(columns=["A", "B"])
    with pytest.raises(TypeError):
        _ = obj.fit(X)


@pytest.mark.koalas
def test_dateformat():
    obj = OrdinalDayOfMonth(columns=["A"])
    with pytest.raises(TypeError):
        OrdinalDayOfMonth(columns=["A"], date_format=0)
    with pytest.raises(ValueError):
        OrdinalDayOfMonth(columns=["A"], date_format="abc")
