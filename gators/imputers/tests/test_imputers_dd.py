# License: Apache-2.0
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.imputers.numerics_imputer import NumericsImputer
from gators.imputers.object_imputer import ObjectImputer


@pytest.fixture()
def data():
    X_int = dd.from_pandas(
        pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]}), npartitions=1
    )
    X_float = dd.from_pandas(
        pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]}),
        npartitions=1,
    )
    X_object = dd.from_pandas(
        pd.DataFrame({"E": ["q", "w", "w", None], "F": ["a", "a", "s", np.nan]}),
        npartitions=1,
    )
    X_int_expected = pd.DataFrame(
        {"A": [0.0, 1.0, 1.0, -9.0], "B": [3.0, 4.0, 4.0, -9.0]}
    )
    X_float_expected = pd.DataFrame(
        {"C": [0.1, 1.1, 2.1, 1.1], "D": [2.1, 3.1, 4.1, 3.1]}
    )
    X_object_expected = pd.DataFrame(
        {"E": ["q", "w", "w", "MISSING"], "F": ["a", "a", "s", "MISSING"]}
    )
    obj_int = NumericsImputer(strategy="constant", value=-9, columns=list("AB")).fit(
        X_int
    )
    obj_float = NumericsImputer(strategy="mean", columns=list("CD")).fit(X_float)
    obj_object = ObjectImputer(strategy="constant", value="MISSING").fit(X_object)
    X_dict = {
        "int": X_int,
        "float": X_float,
        "object": X_object,
    }
    X_expected_dict = {
        "int": X_int_expected,
        "float": X_float_expected,
        "object": X_object_expected,
    }
    objs_dict = {
        "int": obj_int,
        "float": obj_float,
        "object": obj_object,
    }
    return objs_dict, X_dict, X_expected_dict


@pytest.fixture()
def data_num():
    X_int = dd.from_pandas(
        pd.DataFrame(
            {"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]}, dtype=np.float32
        ),
        npartitions=1,
    )
    X_float = dd.from_pandas(
        pd.DataFrame(
            {"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]},
            dtype=np.float32,
        ),
        npartitions=1,
    )
    X_int_expected = pd.DataFrame(
        {"A": [0.0, 1.0, 1.0, -9.0], "B": [3.0, 4.0, 4.0, -9.0]}, dtype=np.float32
    )
    X_float_expected = pd.DataFrame(
        {"C": [0.1, 1.1, 2.1, 1.1], "D": [2.1, 3.1, 4.1, 3.1]}, dtype=np.float32
    )
    obj_int = NumericsImputer(strategy="constant", value=-9, columns=list("AB")).fit(
        X_int
    )
    obj_float = NumericsImputer(strategy="mean", columns=list("CD")).fit(X_float)
    X_dict = {
        "int": X_int,
        "float": X_float,
    }
    X_expected_dict = {
        "int": X_int_expected,
        "float": X_float_expected,
    }
    objs_dict = {
        "int": obj_int,
        "float": obj_float,
    }
    return objs_dict, X_dict, X_expected_dict


@pytest.fixture()
def data_no_missing():
    X_int = dd.from_pandas(
        pd.DataFrame({"A": [0, 1, 1, 8], "B": [3, 4, 4, 8]}, dtype=int), npartitions=1
    )
    X_float = dd.from_pandas(
        pd.DataFrame({"C": [0.1, 1.1, 2.1, 9.0], "D": [2.1, 3.1, 4.1, 9.0]}),
        npartitions=1,
    )
    X_object = dd.from_pandas(
        pd.DataFrame({"E": ["q", "w", "w", "x"], "F": ["a", "a", "s", "x"]}),
        npartitions=1,
    )
    obj_int = NumericsImputer(strategy="constant", value=-9, columns=list("AB")).fit(
        X_int
    )
    obj_float = NumericsImputer(strategy="mean", columns=list("CD")).fit(X_float)
    obj_object = ObjectImputer(strategy="constant", value="MISSING").fit(X_object)
    X_dict = {
        "int": X_int,
        "float": X_float,
        "object": X_object,
    }
    X_expected_dict = {
        "int": X_int.compute().copy(),
        "float": X_float.compute().copy(),
        "object": X_object.compute().copy(),
    }
    objs_dict = {
        "int": obj_int,
        "float": obj_float,
        "object": obj_object,
    }
    return objs_dict, X_dict, X_expected_dict


@pytest.fixture
def data_full():
    X_int = dd.from_pandas(
        pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]}), npartitions=1
    )
    X_float = dd.from_pandas(
        pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]}),
        npartitions=1,
    )
    X_object = dd.from_pandas(
        pd.DataFrame({"E": ["q", "w", "w", np.nan], "F": ["a", "a", "s", None]}),
        npartitions=1,
    )
    X = dd.concat([X_int, X_float, X_object], axis=1)
    X_expected = pd.DataFrame(
        [
            [0.0, 3.0, 0.1, 2.1, "q", "a"],
            [1.0, 4.0, 1.1, 3.1, "w", "a"],
            [1.0, 4.0, 2.1, 4.1, "w", "s"],
            [-9.0, -9.0, 1.1, 3.1, "w", "a"],
        ],
        columns=["A", "B", "C", "D", "E", "F"],
    )
    obj_int = NumericsImputer(strategy="constant", value=-9, columns=list("AB")).fit(X)
    obj_float = NumericsImputer(strategy="mean", columns=list("CD")).fit(X)
    obj_object = ObjectImputer(strategy="most_frequent").fit(X)
    objs_dict = {
        "int": obj_int,
        "float": obj_float,
        "object": obj_object,
    }
    return objs_dict, X, X_expected


def test_int_dd(data):
    objs_dict, X_dict, X_expected_dict = data
    assert_frame_equal(
        objs_dict["int"].transform(X_dict["int"]).compute(), X_expected_dict["int"]
    )


def test_float_dd(data):
    objs_dict, X_dict, X_expected_dict = data
    assert_frame_equal(
        objs_dict["float"].transform(X_dict["float"]).compute(),
        X_expected_dict["float"],
    )


def test_object_dd(data):
    objs_dict, X_dict, X_expected_dict = data
    assert_frame_equal(
        objs_dict["object"].transform(X_dict["object"]).compute(),
        X_expected_dict["object"],
    )


def test_int_dd_np(data):
    objs_dict, X_dict, X_expected_dict = data
    X_new_np = objs_dict["int"].transform_numpy(X_dict["int"].compute().to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["int"].columns)
    assert_frame_equal(X_new, X_expected_dict["int"])


def test_float_dd_np(data):
    objs_dict, X_dict, X_expected_dict = data
    X_new_np = objs_dict["float"].transform_numpy(X_dict["float"].compute().to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["float"].columns)
    assert_frame_equal(X_new, X_expected_dict["float"])


def test_object_dd_np(data):
    objs_dict, X_dict, X_expected_dict = data
    X_new_np = objs_dict["object"].transform_numpy(
        X_dict["object"].compute().to_numpy()
    )
    X_new = pd.DataFrame(X_new_np, columns=X_dict["object"].columns)
    assert_frame_equal(X_new, X_expected_dict["object"])


def test_num_int_dd(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    assert_frame_equal(
        objs_dict["int"]
        .transform(X_dict["int"])
        .compute()
        .astype(np.float32),  # DASK problem
        X_expected_dict["int"],
    )


def test_num_float_dd(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    assert_frame_equal(
        objs_dict["float"]
        .transform(X_dict["float"])
        .compute()
        .astype(np.float32),  # DASK problem
        X_expected_dict["float"],
    )


def test_num_int_dd_np(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    X_new_np = objs_dict["int"].transform_numpy(X_dict["int"].compute().to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["int"].columns)
    assert_frame_equal(X_new, X_expected_dict["int"])


def test_num_float_dd_np(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    X_new_np = objs_dict["float"].transform_numpy(X_dict["float"].compute().to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["float"].columns)
    assert_frame_equal(X_new, X_expected_dict["float"])


def test_no_missing_int_dd(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    assert_frame_equal(
        objs_dict["int"].transform(X_dict["int"]).compute(), X_expected_dict["int"]
    )


def test_no_missing_float_dd(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    assert_frame_equal(
        objs_dict["float"].transform(X_dict["float"]).compute(),
        X_expected_dict["float"],
    )


def test_no_missing_object_dd(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    assert_frame_equal(
        objs_dict["object"].transform(X_dict["object"]).compute(),
        X_expected_dict["object"],
    )


def test_no_missing_int_dd_np(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    X_new_np = objs_dict["int"].transform_numpy(X_dict["int"].compute().to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["int"].columns)
    assert_frame_equal(X_new, X_expected_dict["int"])


def test_no_missing_float_dd_np(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    X_new_np = objs_dict["float"].transform_numpy(X_dict["float"].compute().to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["float"].columns)
    assert_frame_equal(X_new, X_expected_dict["float"])


def test_no_missing_object_dd_np(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    X_new_np = objs_dict["object"].transform_numpy(
        X_dict["object"].compute().to_numpy()
    )
    X_new = pd.DataFrame(X_new_np, columns=X_dict["object"].columns)
    assert_frame_equal(X_new, X_expected_dict["object"])


def test_full_dd(data_full):
    objs_dict, X, X_expected = data_full
    X_new = objs_dict["object"].transform(X)
    X_new = objs_dict["int"].transform(X_new)
    X_new = objs_dict["float"].transform(X_new)
    assert_frame_equal(X_new.compute(), X_expected)


def test_full_dd_np(data_full):
    objs_dict, X, X_expected = data_full
    X_new = objs_dict["object"].transform_numpy(X.compute().to_numpy())
    X_new = objs_dict["int"].transform_numpy(X_new)
    X_new = objs_dict["float"].transform_numpy(X_new)
    X_new = pd.DataFrame(X_new, columns=["A", "B", "C", "D", "E", "F"])
    assert_frame_equal(X_new, X_expected.astype(object))
