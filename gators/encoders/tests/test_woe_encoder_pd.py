# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.encoders.woe_encoder import WOEEncoder


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 0], name="TARGET")
    X_expected = pd.DataFrame(
        {
            "A": {
                0: -1.4350845252893225,
                1: -1.4350845252893225,
                2: -1.4350845252893225,
                3: 1.0216512475319814,
                4: 1.0216512475319814,
                5: 1.0216512475319814,
            },
            "B": {
                0: -1.0986122886681098,
                1: -1.0986122886681098,
                2: 0.5108256237659907,
                3: 0.5108256237659907,
                4: 0.5108256237659907,
                5: 0.5108256237659907,
            },
            "C": {
                0: -0.3364722366212129,
                1: -0.3364722366212129,
                2: -0.3364722366212129,
                3: -0.3364722366212129,
                4: 0.5108256237659907,
                5: 0.5108256237659907,
            },
            "D": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0},
        }
    )
    obj = WOEEncoder(regularization=0.5).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_not_inplace():
    X = pd.DataFrame(
        {
            "A": ["Q", "Q", "Q", "W", "W", "W"],
            "B": ["Q", "Q", "W", "W", "W", "W"],
            "C": ["Q", "Q", "Q", "Q", "W", "W"],
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 0], name="TARGET")
    X_expected = pd.DataFrame(
        {
            "A__woe": {
                0: -1.4350845252893225,
                1: -1.4350845252893225,
                2: -1.4350845252893225,
                3: 1.0216512475319814,
                4: 1.0216512475319814,
                5: 1.0216512475319814,
            },
            "B__woe": {
                0: -1.0986122886681098,
                1: -1.0986122886681098,
                2: 0.5108256237659907,
                3: 0.5108256237659907,
                4: 0.5108256237659907,
                5: 0.5108256237659907,
            },
            "C__woe": {
                0: -0.3364722366212129,
                1: -0.3364722366212129,
                2: -0.3364722366212129,
                3: -0.3364722366212129,
                4: 0.5108256237659907,
                5: 0.5108256237659907,
            },
        }
    )
    X_expected = pd.concat([X, X_expected], axis=1)
    obj = WOEEncoder(regularization=0.5, inplace=False).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_no_cat():
    X = pd.DataFrame(np.zeros((6, 3)), columns=list("ABC"))
    y = pd.Series([0, 0, 0, 1, 1, 0], name="TARGET")
    obj = WOEEncoder().fit(X, y)
    return obj, X, X.copy()


def test_pd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_pd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_pd(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_no_cat_pd_np(data_no_cat):
    obj, X, X_expected = data_no_cat
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected)


def test_data_not_inplace_pd(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_new = obj.transform(X)
    assert_frame_equal(X_new, X_expected)


def test_data_not_inplace_pd_np(data_not_inplace):
    obj, X, X_expected = data_not_inplace
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))


def test_display_mapping(data_not_inplace):
    obj, _, _ = data_not_inplace
    obj.display_mapping(cmap="Reds")
    assert True


def test_init():
    with pytest.raises(TypeError):
        _ = WOEEncoder(inplace="yes")
    with pytest.raises(TypeError):
        _ = WOEEncoder(regularization="a")
    with pytest.raises(ValueError):
        _ = WOEEncoder(regularization=-1)


def test_display_mapping_input(data_not_inplace):
    obj, _, _ = data_not_inplace
    with pytest.raises(TypeError):
        obj.display_mapping(cmap="Reds", decimal=-1)
    with pytest.raises(TypeError):
        obj.display_mapping(cmap="Reds", decimal=1.1)
    with pytest.raises(TypeError):
        obj.display_mapping(cmap="Reds", title=0)
    with pytest.raises(TypeError):
        obj.display_mapping(cmap="Reds", k=0)
    with pytest.raises(TypeError):
        obj.display_mapping(cmap="Reds", k=1.5)
