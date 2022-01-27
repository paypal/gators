# License: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gators.imputers.numerics_imputer import NumericsImputer
from gators.imputers.object_imputer import ObjectImputer


@pytest.fixture()
def data():
    X_int = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", None], "F": ["a", "a", "s", np.nan]})
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
    X_int = pd.DataFrame(
        {"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]}, dtype=np.float32
    )
    X_float = pd.DataFrame(
        {"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]}, dtype=np.float32
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
    X_int = pd.DataFrame({"A": [0, 1, 1, 8], "B": [3, 4, 4, 8]}, dtype=int)
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, 9.0], "D": [2.1, 3.1, 4.1, 9.0]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", "x"], "F": ["a", "a", "s", "x"]})
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
        "int": X_int.copy(),
        "float": X_float.copy(),
        "object": X_object.copy(),
    }
    objs_dict = {
        "int": obj_int,
        "float": obj_float,
        "object": obj_object,
    }
    return objs_dict, X_dict, X_expected_dict


@pytest.fixture
def data_full():
    X_int = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", np.nan], "F": ["a", "a", "s", None]})
    X = pd.concat([X_int, X_float, X_object], axis=1)
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
    obj_float = NumericsImputer(strategy="median", columns=list("CD")).fit(X)
    obj_object = ObjectImputer(strategy="most_frequent").fit(X)
    objs_dict = {
        "int": obj_int,
        "float": obj_float,
        "object": obj_object,
    }
    return objs_dict, X, X_expected


def test_int_pd(data):
    objs_dict, X_dict, X_expected_dict = data
    assert_frame_equal(
        objs_dict["int"].transform(X_dict["int"]),
        X_expected_dict["int"],
    )


def test_float_pd(data):
    objs_dict, X_dict, X_expected_dict = data
    assert_frame_equal(
        objs_dict["float"].transform(X_dict["float"]),
        X_expected_dict["float"],
    )


def test_object_pd(data):
    objs_dict, X_dict, X_expected_dict = data
    assert_frame_equal(
        objs_dict["object"].transform(X_dict["object"]),
        X_expected_dict["object"],
    )


def test_int_pd_np(data):
    objs_dict, X_dict, X_expected_dict = data
    X_new_np = objs_dict["int"].transform_numpy(X_dict["int"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["int"].columns)
    assert_frame_equal(X_new, X_expected_dict["int"])


def test_float_pd_np(data):
    objs_dict, X_dict, X_expected_dict = data
    X_new_np = objs_dict["float"].transform_numpy(X_dict["float"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["float"].columns)
    assert_frame_equal(X_new, X_expected_dict["float"])


def test_object_pd_np(data):
    objs_dict, X_dict, X_expected_dict = data
    X_new_np = objs_dict["object"].transform_numpy(X_dict["object"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["object"].columns)
    assert_frame_equal(X_new, X_expected_dict["object"])


def test_num_int_pd(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    assert_frame_equal(
        objs_dict["int"].transform(X_dict["int"]),
        X_expected_dict["int"],
    )


def test_num_float_pd(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    assert_frame_equal(
        objs_dict["float"].transform(X_dict["float"]),
        X_expected_dict["float"],
    )


def test_num_int_pd_np(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    X_new_np = objs_dict["int"].transform_numpy(X_dict["int"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["int"].columns)
    assert_frame_equal(X_new, X_expected_dict["int"])


def test_num_float_pd_np(data_num):
    objs_dict, X_dict, X_expected_dict = data_num
    X_new_np = objs_dict["float"].transform_numpy(X_dict["float"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["float"].columns)
    assert_frame_equal(X_new, X_expected_dict["float"])


def test_no_missing_int_pd(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    assert_frame_equal(
        objs_dict["int"].transform(X_dict["int"]),
        X_expected_dict["int"],
    )


def test_no_missing_float_pd(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    assert_frame_equal(
        objs_dict["float"].transform(X_dict["float"]),
        X_expected_dict["float"],
    )


def test_no_missing_object_pd(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    assert_frame_equal(
        objs_dict["object"].transform(X_dict["object"]),
        X_expected_dict["object"],
    )


def test_no_missing_int_pd_np(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    X_new_np = objs_dict["int"].transform_numpy(X_dict["int"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["int"].columns)
    assert_frame_equal(X_new, X_expected_dict["int"])


def test_no_missing_float_pd_np(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    X_new_np = objs_dict["float"].transform_numpy(X_dict["float"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["float"].columns)
    assert_frame_equal(X_new, X_expected_dict["float"])


def test_no_missing_object_pd_np(data_no_missing):
    objs_dict, X_dict, X_expected_dict = data_no_missing
    X_new_np = objs_dict["object"].transform_numpy(X_dict["object"].to_numpy())
    X_new = pd.DataFrame(X_new_np, columns=X_dict["object"].columns)
    assert_frame_equal(X_new, X_expected_dict["object"])


def test_full_pd(data_full):
    objs_dict, X, X_expected = data_full
    X_new = objs_dict["object"].transform(X)
    X_new = objs_dict["int"].transform(X_new)
    X_new = objs_dict["float"].transform(X_new)
    assert_frame_equal(X_new, X_expected)


def test_full_pd_np(data_full):
    objs_dict, X, X_expected = data_full
    X_new = objs_dict["object"].transform_numpy(X.to_numpy())
    X_new = objs_dict["int"].transform_numpy(X_new)
    X_new = objs_dict["float"].transform_numpy(X_new)
    X_new = pd.DataFrame(X_new, columns=["A", "B", "C", "D", "E", "F"])
    assert_frame_equal(X_new, X_expected.astype(object))


def test_imputers_columns_pd():
    X_int = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", np.nan], "F": ["a", "a", "s", None]})
    X = pd.concat([X_int, X_float, X_object], axis=1)
    X_expected = pd.DataFrame(
        [
            [0.0, 3.0, 0.1, 2.1, "q", "a"],
            [1.0, 4.0, 1.1, 3.1, "w", "a"],
            [1.0, 4.0, 2.1, 4.1, "w", "s"],
            [-9.0, -99.0, -999.0, -9999.0, "missing", "MISSING"],
        ],
        columns=["A", "B", "C", "D", "E", "F"],
    )
    obj_int_A = NumericsImputer(strategy="constant", value=-9, columns=["A"]).fit(X)
    obj_int_B = NumericsImputer(strategy="constant", value=-99, columns=["B"]).fit(X)
    obj_float_C = NumericsImputer(strategy="constant", value=-999.0, columns=["C"]).fit(
        X
    )
    obj_float_D = NumericsImputer(
        strategy="constant", value=-9999.0, columns=["D"]
    ).fit(X)
    obj_object_E = ObjectImputer(
        strategy="constant", value="missing", columns=["E"]
    ).fit(X)
    obj_object_F = ObjectImputer(
        strategy="constant", value="MISSING", columns=["F"]
    ).fit(X)
    X_new = obj_int_A.transform(X)
    X_new = obj_int_B.transform(X_new)
    X_new = obj_float_C.transform(X_new)
    X_new = obj_float_D.transform(X_new)
    X_new = obj_object_E.transform(X_new)
    X_new = obj_object_F.transform(X_new)
    assert_frame_equal(X_new, X_expected)


def test_imputers_columns_pd_np():
    X_int = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", np.nan], "F": ["a", "a", "s", None]})
    X = pd.concat([X_int, X_float, X_object], axis=1)
    X_expected = pd.DataFrame(
        [
            [0.0, 3.0, 0.1, 2.1, "q", "a"],
            [1.0, 4.0, 1.1, 3.1, "w", "a"],
            [1.0, 4.0, 2.1, 4.1, "w", "s"],
            [-9.0, -99.0, -999.0, -9999.0, "missing", "MISSING"],
        ],
        columns=["A", "B", "C", "D", "E", "F"],
    )
    obj_int_A = NumericsImputer(strategy="constant", value=-9, columns=["A"]).fit(X)
    obj_int_B = NumericsImputer(strategy="constant", value=-99, columns=["B"]).fit(X)
    obj_float_C = NumericsImputer(strategy="constant", value=-999.0, columns=["C"]).fit(
        X
    )
    obj_float_D = NumericsImputer(
        strategy="constant", value=-9999.0, columns=["D"]
    ).fit(X)
    obj_object_E = ObjectImputer(
        strategy="constant", value="missing", columns=["E"]
    ).fit(X)
    obj_object_F = ObjectImputer(
        strategy="constant", value="MISSING", columns=["F"]
    ).fit(X)
    X_new = obj_int_A.transform_numpy(X.to_numpy())
    X_new = obj_int_B.transform_numpy(X_new)
    X_new = obj_float_C.transform_numpy(X_new)
    X_new = obj_float_D.transform_numpy(X_new)
    X_new = obj_object_E.transform_numpy(X_new)
    X_new = obj_object_F.transform_numpy(X_new)
    assert_frame_equal(
        pd.DataFrame(X_new, columns=list("ABCDEF")), X_expected.astype(object)
    )


def test_imputers_num_pd():
    X_int = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", np.nan], "F": ["a", "a", "s", None]})
    X = pd.concat([X_int, X_float, X_object], axis=1)
    X_expected = pd.DataFrame(
        [
            [0.0, 3.0, 0.1, 2.1, "q", "a"],
            [1.0, 4.0, 1.1, 3.1, "w", "a"],
            [1.0, 4.0, 2.1, 4.1, "w", "s"],
            [-9.0, -9.0, -9.0, -9.0, "MISSING", "MISSING"],
        ],
        columns=["A", "B", "C", "D", "E", "F"],
    )
    obj_num = NumericsImputer(strategy="constant", value=-9.0).fit(X)
    obj_object = ObjectImputer(strategy="constant", value="MISSING").fit(X)
    X_new = obj_num.transform(X)
    X_new = obj_object.transform(X_new)
    assert_frame_equal(X_new, X_expected)


def test_imputers_num_pd_np():
    X_int = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    X_float = pd.DataFrame({"C": [0.1, 1.1, 2.1, np.nan], "D": [2.1, 3.1, 4.1, np.nan]})
    X_object = pd.DataFrame({"E": ["q", "w", "w", np.nan], "F": ["a", "a", "s", None]})
    X = pd.concat([X_int, X_float, X_object], axis=1)
    X_expected = pd.DataFrame(
        [
            [0.0, 3.0, 0.1, 2.1, "q", "a"],
            [1.0, 4.0, 1.1, 3.1, "w", "a"],
            [1.0, 4.0, 2.1, 4.1, "w", "s"],
            [-9.0, -9.0, -9.0, -9.0, "MISSING", "MISSING"],
        ],
        columns=["A", "B", "C", "D", "E", "F"],
    )
    obj_num = NumericsImputer(strategy="constant", value=-9.0).fit(X)
    obj_object = ObjectImputer(strategy="constant", value="MISSING").fit(X)
    X_new = obj_num.transform_numpy(X.to_numpy())
    X_new = obj_object.transform_numpy(X_new)
    assert_frame_equal(
        pd.DataFrame(X_new, columns=list("ABCDEF")), X_expected.astype(object)
    )


def test_num_np():
    X = pd.DataFrame({"A": [0, 1, np.nan]})
    obj = NumericsImputer(strategy="mean").fit(X)
    assert obj.transform_numpy(X.to_numpy()).tolist() == [[0.0], [1.0], [0.5]]


def test_imputers_stategy():
    X = pd.DataFrame([])
    with pytest.raises(TypeError):
        _ = NumericsImputer(strategy=0)
    with pytest.raises(TypeError):
        _ = NumericsImputer(strategy=0)
    with pytest.raises(TypeError):
        _ = NumericsImputer(strategy="constant", value="a").fit(X)
    with pytest.raises(TypeError):
        _ = ObjectImputer(strategy="constant", value=1).fit(X)
    with pytest.raises(ValueError):
        _ = NumericsImputer(strategy="").fit(X)
    with pytest.raises(ValueError):
        _ = ObjectImputer(strategy="").fit(X)
    with pytest.raises(ValueError):
        _ = ObjectImputer(strategy="mean").fit(X)
    with pytest.raises(ValueError):
        _ = ObjectImputer(strategy="median").fit(X)
    with pytest.raises(ValueError):
        _ = ObjectImputer(strategy="constant").fit(X)
    with pytest.raises(ValueError):
        _ = NumericsImputer(strategy="constant").fit(X)
    with pytest.raises(ValueError):
        _ = NumericsImputer(strategy="abc").fit(X)


def test_compute_stategy():
    with pytest.raises(ValueError):
        X = pd.DataFrame(np.arange(9).reshape(3, 3) + 0.1, columns=list("qwe"))
        X.iloc[:, 0] = np.nan
        _ = NumericsImputer(strategy="mean").fit(X)


def test_imputers_input_data():
    with pytest.raises(TypeError):
        _ = ObjectImputer(strategy="most_frequent").fit(np.array([[]]))
    with pytest.raises(TypeError):
        _ = ObjectImputer(strategy="most_frequent", columns="a")


def test_warnings_empty_columns(data):
    _, X_dict, _ = data
    with pytest.warns(Warning):
        obj = ObjectImputer(strategy="most_frequent")
        obj.fit(X_dict["int"])
    with pytest.warns(Warning):
        obj = NumericsImputer(strategy="mean")
        obj.fit(X_dict["object"])


def test_empty_columns_object():
    X = pd.DataFrame({"A": [0, 1, 1, np.nan], "B": [3, 4, 4, np.nan]})
    obj = ObjectImputer(strategy="most_frequent")
    _ = obj.fit(X)
    assert_frame_equal(obj.fit_transform(X.copy()), X)
    assert_frame_equal(
        pd.DataFrame(obj.transform_numpy(X.to_numpy())), pd.DataFrame(X.to_numpy())
    )


def test_num_idx_columns_empty():
    X = pd.DataFrame({"A": ["a", "b", "b", "c"]})
    obj = NumericsImputer(strategy="mean").fit(X)
    _ = obj.fit(X)
    assert_frame_equal(obj.transform(X.copy()), X)
    assert_frame_equal(
        pd.DataFrame(obj.transform_numpy(X.to_numpy())), pd.DataFrame(X.to_numpy())
    )
