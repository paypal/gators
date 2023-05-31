# License: Apache-2.0
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.scalers.yeo_johnson import YeoJohnson


@pytest.fixture
def data():

    X = pd.DataFrame(
        {
            "A": {0: 3.0, 1: 1.0, 2: -3.0, 3: -1.0, 4: -3.0},
            "B": {0: 22.0, 1: 38.0, 2: -26.0, 3: 35.0, 4: 3 - 5.0},
            "C": {0: 7.25, 1: 71.2833, 2: -7.925, 3: -53.1, 4: -8.05},
        }
    )
    lambdas_dict = {
        "A": 0.8130050344716966,
        "B": 1.0431595843133055,
        "C": 0.9168245659045446,
    }
    X_expected = pd.DataFrame(
        {
            "A": {
                0: 2.566505499508099,
                1: 0.9309500374418126,
                2: -3.52463810254087,
                3: -1.0756407146577545,
                4: -3.52463810254087,
            },
            "B": {
                0: 24.28482282762873,
                1: 42.83224067680155,
                2: -23.431273698291413,
                3: 39.324309978116744,
                4: -1.9450189509681384,
            },
            "C": {
                0: 6.459179705999775,
                1: 54.13297098278756,
                2: -8.961788539290808,
                3: -68.6845869657268,
                4: -9.111836147073353,
            },
        },
    )

    return YeoJohnson(lambdas_dict=lambdas_dict).fit(X), X, X_expected


@pytest.fixture
def data_not_inplace():

    X = pd.DataFrame(
        {
            "A": {0: 3.0, 1: 1.0, 2: -3.0, 3: -1.0, 4: -3.0},
            "B": {0: 22.0, 1: 38.0, 2: -26.0, 3: 35.0, 4: 3 - 5.0},
            "C": {0: 7.25, 1: 71.2833, 2: -7.925, 3: -53.1, 4: -8.05},
        }
    )
    lambdas_dict = {"A": 2, "B": 2, "C": 2}
    X_expected = pd.DataFrame(
        {
            "A": {0: 3.0, 1: 1.0, 2: -3.0, 3: -1.0, 4: -3.0},
            "B": {0: 22.0, 1: 38.0, 2: -26.0, 3: 35.0, 4: 3 - 5.0},
            "C": {0: 7.25, 1: 71.2833, 2: -7.925, 3: -53.1, 4: -8.05},
            "A__yeojohnson": {
                0: 7.5,
                1: 1.5,
                2: -1.3862943611198906,
                3: -0.6931471805599453,
                4: -1.3862943611198906,
            },
            "B__yeojohnson": {
                0: 264.0,
                1: 760.0,
                2: -3.295836866004329,
                3: 647.5,
                4: -1.0986122886681098,
            },
            "C__yeojohnson": {
                0: 33.53125,
                1: 2611.9377294449996,
                2: -2.188856327665703,
                3: -3.9908341858524357,
                4: -2.2027647577118348,
            },
        }
    )

    return YeoJohnson(lambdas_dict=lambdas_dict, inplace=False).fit(X), X, X_expected


@pytest.fixture
def data_0lambda():
    X = pd.DataFrame(
        {
            "A": {0: 3.0, 1: 1.0, 2: -3.0, 3: -1.0, 4: -3.0},
            "B": {0: 22.0, 1: 38.0, 2: -26.0, 3: 35.0, 4: 3 - 5.0},
            "C": {0: 7.25, 1: 71.2833, 2: -7.925, 3: -53.1, 4: -8.05},
        }
    )
    lambdas_dict = {"A": 0, "B": 0, "C": 0}
    X_expected = pd.DataFrame(
        {
            "A": {
                0: 1.3862943611198906,
                1: 0.6931471805599453,
                2: -7.5,
                3: -1.5,
                4: -7.5,
            },
            "B": {
                0: 3.1354942159291497,
                1: 3.6635616461296463,
                2: -364.0,
                3: 3.58351893845611,
                4: -4.0,
            },
            "C": {
                0: 2.1102132003465894,
                1: 4.2805931204649,
                2: -39.32781250000001,
                3: -1462.905,
                4: -40.45125000000001,
            },
        }
    )

    return YeoJohnson(lambdas_dict=lambdas_dict).fit(X), X, X_expected


@pytest.fixture
def data_2lambda():

    X = pd.DataFrame(
        {
            "A": {0: 3.0, 1: 1.0, 2: -3.0, 3: -1.0, 4: -3.0},
            "B": {0: 22.0, 1: 38.0, 2: -26.0, 3: 35.0, 4: 3 - 5.0},
            "C": {0: 7.25, 1: 71.2833, 2: -7.925, 3: -53.1, 4: -8.05},
        }
    )
    lambdas_dict = {"A": 2, "B": 2, "C": 2}
    X_expected = pd.DataFrame(
        {
            "A": {
                0: 7.5,
                1: 1.5,
                2: -1.3862943611198906,
                3: -0.6931471805599453,
                4: -1.3862943611198906,
            },
            "B": {
                0: 264.0,
                1: 760.0,
                2: -3.295836866004329,
                3: 647.5,
                4: -1.0986122886681098,
            },
            "C": {
                0: 33.53125,
                1: 2611.9377294449996,
                2: -2.188856327665703,
                3: -3.9908341858524357,
                4: -2.2027647577118348,
            },
        }
    )

    return YeoJohnson(lambdas_dict=lambdas_dict).fit(X), X, X_expected


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_0lambda_pd(data_0lambda):
    obj, X, X_expected = data_0lambda
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_2lambda_pd(data_2lambda):
    obj, X, X_expected = data_2lambda
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_not_inplace_pd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_0lambda_pd_np(data_0lambda):
    obj, X, X_expected = data_0lambda
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_2lambda_pd_np(data_2lambda):
    obj, X, X_expected = data_2lambda
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_not_inplace_pd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_inputs():
    with pytest.raises(TypeError):
        YeoJohnson(lambdas_dict=[])
    with pytest.raises(ValueError):
        YeoJohnson(lambdas_dict={})
    with pytest.raises(TypeError):
        YeoJohnson(lambdas_dict={"A": 0.5}, inplace="x")