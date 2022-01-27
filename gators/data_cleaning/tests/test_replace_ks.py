import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.replace import Replace


@pytest.fixture
def data_ks():
    X = ks.DataFrame(
        {
            "A": list("abcd"),
            "B": list("abcd"),
            "C": [0, 1, 2, 3],
        }
    )
    X_expected = pd.DataFrame(
        {
            "A": ["W", "X", "c", "d"],
            "B": ["a", "b", "Y", "Z"],
            "C": [0, 1, 2, 3],
        }
    )
    to_replace_dict = {"A": {"a": "W", "b": "X"}, "B": {"c": "Y", "d": "Z"}}
    obj = Replace(to_replace_dict=to_replace_dict).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, X_expected = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
