# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from gators.util.iv import compute_iv


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "Embarked": {0: "S", 1: "C", 2: "S", 3: "S", 4: "S"},
            "Sex": {
                0: "male",
                1: "female",
                2: "female",
                3: "female",
                4: "male",
            },
            "Ticket": {
                0: "A",
                1: "B",
                2: "C",
                3: "D",
                4: "A",
            },
        }
    )

    y = pd.Series({0: 0, 1: 1, 2: 1, 3: 1, 4: 0}, name="TARGET")
    stats_expected = pd.DataFrame(
        {
            "0": {
                ("Embarked", "C"): 0,
                ("Embarked", "S"): 2,
                ("Sex", "female"): 0,
                ("Sex", "male"): 2,
                ("Ticket", "A"): 2,
                ("Ticket", "B"): 0,
                ("Ticket", "C"): 0,
                ("Ticket", "D"): 0,
            },
            "1": {
                ("Embarked", "C"): 1,
                ("Embarked", "S"): 2,
                ("Sex", "female"): 3,
                ("Sex", "male"): 0,
                ("Ticket", "A"): 0,
                ("Ticket", "B"): 1,
                ("Ticket", "C"): 1,
                ("Ticket", "D"): 1,
            },
            "distrib_0": {
                ("Embarked", "C"): 0.045454545454545456,
                ("Embarked", "S"): 0.9545454545454545,
                ("Sex", "female"): 0.045454545454545456,
                ("Sex", "male"): 0.9545454545454545,
                ("Ticket", "A"): 0.9545454545454545,
                ("Ticket", "B"): 0.045454545454545456,
                ("Ticket", "C"): 0.045454545454545456,
                ("Ticket", "D"): 0.045454545454545456,
            },
            "distrib_1": {
                ("Embarked", "C"): 0.34375,
                ("Embarked", "S"): 0.65625,
                ("Sex", "female"): 0.96875,
                ("Sex", "male"): 0.03125,
                ("Ticket", "A"): 0.03125,
                ("Ticket", "B"): 0.34375,
                ("Ticket", "C"): 0.34375,
                ("Ticket", "D"): 0.34375,
            },
            "woe": {
                ("Embarked", "C"): 2.0232018233569597,
                ("Embarked", "S"): -0.3746934494414107,
                ("Sex", "female"): 3.0592937550437354,
                ("Sex", "male"): -3.4192158871648335,
                ("Ticket", "A"): -3.4192158871648335,
                ("Ticket", "B"): 2.0232018233569597,
                ("Ticket", "C"): 2.0232018233569597,
                ("Ticket", "D"): 2.0232018233569597,
            },
        },
    )
    stats_expected.index.names = ["variable", "value"]
    iv_expected = pd.Series(
        {
            "Embarked": 0.7152812603517865,
            "Sex": 5.981578504880071,
            "Ticket": 4.967482209335264,
        }
    )
    iv_expected.name = "iv"
    return X, y, stats_expected, iv_expected


def test_stats(data):
    X, y, stats_expected, _ = data
    _, stats = compute_iv(X, y)
    assert_frame_equal(stats, stats_expected)


def test_iv(data):
    X, y, _, iv_expected = data
    iv, _ = compute_iv(X, y)
    assert_series_equal(iv, iv_expected)
