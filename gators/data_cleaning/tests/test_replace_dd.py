import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.replace import Replace


@pytest.fixture
def data():
    X = dd.from_pandas(
        pd.DataFrame(
            {
                "A": list("abcd"),
                "B": list("abcd"),
                "C": [0, 1, 2, 3],
            }
        ),
        npartitions=1,
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


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute()
    X_new[["A", "B"]] = X_new[["A", "B"]].astype(object)
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, X_expected = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    X_new = pd.DataFrame(X_numpy_new, columns=X_expected.columns)
    assert_frame_equal(X_new, X_expected.astype(object))
