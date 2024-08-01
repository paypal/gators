# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest

from gators.sampling import SupervisedSampling


@pytest.fixture
def data():
    n_rows = 14
    n_cols = 5
    n_samples = 7
    np.random.seed(1)
    X = pd.DataFrame(
        np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=list("ABCDE")
    )
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2], name="TARGET")
    obj = SupervisedSampling(frac_dict={0: 0.2, 1: 0.5, 2: 1.0})
    X_expected_shape = (5, 5)
    y_expected_shape = (5,)
    return obj, X, y, X_expected_shape, y_expected_shape


def test_pd(data):
    obj, X, y, X_expected_shape, y_expected_shape = data
    X_new, y_new = obj.transform(X, y)
    assert X_new.shape == X_expected_shape
    assert y_new.shape == y_expected_shape
    assert (y_new == 0).sum() == 2
    assert (y_new == 1).sum() == 1
    assert (y_new == 2).sum() == 2


def test_init():
    with pytest.raises(TypeError):
        _ = SupervisedSampling(frac_dict="a")
    with pytest.raises(TypeError):
        _ = SupervisedSampling(frac_dict={"a": 0.1})
    with pytest.raises(TypeError):
        _ = SupervisedSampling(frac_dict={1: "a"})
    with pytest.raises(TypeError):
        _ = SupervisedSampling(frac_dict={1: 0.1}, random_state=-1)
