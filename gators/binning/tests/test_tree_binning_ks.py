# License: Apache-2.0
import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ks.set_option("compute.default_index_type", "distributed-sequence")

from gators.binning import TreeBinning


@pytest.fixture
def data():
    X = ks.DataFrame(
        {
            "A": [1.07, -2.59, -1.54, 1.72],
            "B": [-1.19, -0.22, -0.28, 1.28],
            "C": [-1.15, 1.92, 1.09, -0.95],
            "D": ["a", "b", "c", "d"],
        }
    )
    y = ks.Series([0, 1, 0, 1], name="TARGET")
    X_expected = pd.DataFrame(
        {
            "A": [1.07, -2.59, -1.54, 1.72],
            "B": [-1.19, -0.22, -0.28, 1.28],
            "C": [-1.15, 1.92, 1.09, -0.95],
            "D": ["a", "b", "c", "d"],
            "A__bin": ["[-2.06, 1.4)", "(-inf, -2.06)", "[-2.06, 1.4)", "[1.4, inf)"],
            "B__bin": [
                "(-inf, -0.25)",
                "[-0.25, inf)",
                "(-inf, -0.25)",
                "[-0.25, inf)",
            ],
            "C__bin": ["(-inf, -1.05)", "[0.07, inf)", "[0.07, inf)", "[-1.05, 0.07)"],
        }
    )
    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    obj = TreeBinning(tree=tree).fit(X, y)
    return obj, X, X_expected


@pytest.fixture
def data_regression():
    max_depth = 2
    X = ks.DataFrame(
        {
            "A": [-0.1, 1.45, 0.98, -0.98],
            "B": [-0.15, 0.14, 0.4, 1.87],
            "C": [0.95, 0.41, 1.76, 2.24],
        }
    )
    y = ks.Series([39.9596835, 36.65644911, 137.24445075, 300.15325913], name="TARGET")
    X_expected = pd.DataFrame(
        {
            "A": ["[-0.54, 1.22)", "[1.22, inf)", "[-0.54, 1.22)", "(-inf, -0.54)"],
            "B": ["(-inf, 0.27)", "(-inf, 0.27)", "[0.27, 1.14)", "[1.14, inf)"],
            "C": ["(-inf, 1.35)", "(-inf, 1.35)", "[1.35, 2.0)", "[2.0, inf)"],
        }
    )
    tree = DecisionTreeRegressor(max_depth=2, random_state=0)
    obj = TreeBinning(tree=tree, inplace=True).fit(X, y)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data):
    obj, X, X_expected = data
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))


@pytest.mark.koalas
def test_regression_ks(data_regression):
    obj, X, X_expected = data_regression
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_regression_ks_np(data_regression):
    obj, X, X_expected = data_regression
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(
        X_numpy_new, columns=X_expected.columns, index=X_expected.index
    )
    assert_frame_equal(X_new, X_expected.astype(object))
