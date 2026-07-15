"""Tests for HashEncoder."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.encoders import HashEncoder


@pytest.fixture
def X():
    return pl.DataFrame(
        {
            "color": ["red", "blue", "green", "red", "blue"],
            "size": ["S", "M", "L", "XL", "S"],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_auto_detect_subset_skips_numeric(X):
    enc = HashEncoder(n_features=8)
    enc.fit(X)
    assert set(enc.subset) == {"color", "size"}
    assert "weight" not in enc.subset


def test_output_columns_inplace_false(X):
    enc = HashEncoder(n_features=8, inplace=False, drop_columns=True)
    enc.fit(X)
    result = enc.transform(X)
    assert "color__hash" in result.columns
    assert "size__hash" in result.columns
    assert "color" not in result.columns
    assert "weight" in result.columns  # numeric untouched


def test_output_columns_inplace_true(X):
    enc = HashEncoder(n_features=8, inplace=True)
    enc.fit(X)
    result = enc.transform(X)
    assert "color" in result.columns
    assert "color__hash" not in result.columns
    assert result["color"].dtype == pl.Float64


def test_output_columns_inplace_false_drop_false(X):
    enc = HashEncoder(n_features=8, inplace=False, drop_columns=False)
    enc.fit(X)
    result = enc.transform(X)
    assert "color" in result.columns
    assert "color__hash" in result.columns


def test_hash_values_in_range(X):
    n = 8
    enc = HashEncoder(n_features=n, inplace=False)
    enc.fit(X)
    result = enc.transform(X)
    for col in ["color__hash", "size__hash"]:
        assert (result[col] >= 0).all()
        assert (result[col] < n).all()


def test_deterministic_output(X):
    enc = HashEncoder(n_features=16, inplace=False)
    enc.fit(X)
    r1 = enc.transform(X)
    r2 = enc.transform(X)
    assert_frame_equal(r1, r2)


def test_unknown_category_handled(X):
    enc = HashEncoder(n_features=8, inplace=False)
    enc.fit(X)
    X_new = pl.DataFrame(
        {"color": ["purple", "neon"], "size": ["XXL", "XXXL"], "weight": [1.0, 2.0]}
    )
    # Should not raise; unseen values hash into [0, n_features)
    result = enc.transform(X_new)
    assert (result["color__hash"] >= 0).all()
    assert (result["color__hash"] < 8).all()


def test_same_value_same_hash(X):
    enc = HashEncoder(n_features=32, inplace=False)
    enc.fit(X)
    result = enc.transform(X)
    # "red" appears at rows 0 and 3 — must produce identical hash
    assert result["color__hash"][0] == result["color__hash"][3]
    # "S" appears at rows 0 and 4 — same hash
    assert result["size__hash"][0] == result["size__hash"][4]


def test_subset_parameter(X):
    enc = HashEncoder(n_features=8, subset=["color"], inplace=False)
    enc.fit(X)
    result = enc.transform(X)
    assert "color__hash" in result.columns
    assert "size__hash" not in result.columns


def test_invalid_n_features():
    with pytest.raises(Exception):
        HashEncoder(n_features=1)  # must be >= 2


def test_get_params():
    enc = HashEncoder(n_features=32, drop_columns=False)
    params = enc.get_params()
    assert params["n_features"] == 32
    assert params["drop_columns"] is False


def test_set_params():
    enc = HashEncoder()
    enc.set_params(n_features=64)
    assert enc.n_features == 64


def test_numeric_only_dataframe_no_columns_encoded():
    X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    enc = HashEncoder(n_features=8)
    enc.fit(X)
    result = enc.transform(X)
    # No categorical cols → output identical to input
    assert_frame_equal(result, X)


def test_hash_output_dtype_float64(X):
    enc = HashEncoder(n_features=8, inplace=False)
    enc.fit(X)
    result = enc.transform(X)
    assert result["color__hash"].dtype == pl.Float64
    assert result["size__hash"].dtype == pl.Float64
