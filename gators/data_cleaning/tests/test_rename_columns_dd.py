import dask.dataframe as dd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.rename_columns import RenameColumns


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
            "A_f": list("abcd"),
            "B_f": list("abcd"),
            "C": [0, 1, 2, 3],
        }
    )
    to_rename_dict = {"A": "A_f", "B": "B_f"}
    obj = RenameColumns(to_rename_dict=to_rename_dict).fit(X)
    return obj, X, X_expected


def test_dd(data):
    obj, X, X_expected = data
    X_new = obj.transform(X).compute()
    X_new[["A_f", "B_f"]] = X_new[["A_f", "B_f"]].astype(object)
    assert_frame_equal(X_new, X_expected)


def test_dd_np(data):
    obj, X, _ = data
    X_numpy_new = obj.transform_numpy(X.compute().to_numpy())
    assert X_numpy_new.tolist() == X.compute().to_numpy().tolist()
