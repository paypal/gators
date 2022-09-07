import databricks.koalas as ks
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.data_cleaning.rename_columns import RenameColumns


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
            "A_f": list("abcd"),
            "B_f": list("abcd"),
            "C": [0, 1, 2, 3],
        }
    )
    to_rename_dict = {"A": "A_f", "B": "B_f"}
    obj = RenameColumns(to_rename_dict=to_rename_dict).fit(X)
    return obj, X, X_expected


@pytest.mark.koalas
def test_ks(data_ks):
    obj, X, X_expected = data_ks
    X_new = obj.transform(X)
    assert_frame_equal(X_new.to_pandas(), X_expected)


@pytest.mark.koalas
def test_ks_np(data_ks):
    obj, X, _ = data_ks
    X_numpy_new = obj.transform_numpy(X.to_numpy())
    assert X_numpy_new.tolist() == X.to_numpy().tolist()
