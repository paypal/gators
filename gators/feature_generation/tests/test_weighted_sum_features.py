import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import WeightedSumFeatures


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": [10.0, 20.0, 30.0],
            "other": [99.0, 99.0, 99.0],
        }
    )


# ---------------------------------------------------------------------------
# Basic transform — single group, equal weights
# ---------------------------------------------------------------------------


def test_transform_single_group_two_columns(sample_df):
    # equal weights, no bias: a + b
    transformer = WeightedSumFeatures(column_groups=[["a", "b"]])
    result = transformer.fit_transform(sample_df)

    expected_vals = [5.0, 7.0, 9.0]
    assert result["a__b__wsum"].to_list() == pytest.approx(expected_vals)


def test_transform_single_group_single_column(sample_df):
    # identity: 1*a
    transformer = WeightedSumFeatures(column_groups=[["a"]])
    result = transformer.fit_transform(sample_df)

    assert result["a__wsum"].to_list() == pytest.approx([1.0, 2.0, 3.0])


def test_output_dtype_is_float64():
    X = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    transformer = WeightedSumFeatures(column_groups=[["x", "y"]])
    result = transformer.fit_transform(X)

    assert result["x__y__wsum"].dtype == pl.Float64


def test_auto_generated_column_name(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a", "b", "c"]])
    result = transformer.fit_transform(sample_df)

    assert "a__b__c__wsum" in result.columns


# ---------------------------------------------------------------------------
# Custom coefficients
# ---------------------------------------------------------------------------


def test_transform_custom_coefficients(sample_df):
    # 2*a + 3*b
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        coefficients=[[2.0, 3.0]],
    )
    result = transformer.fit_transform(sample_df)

    expected = [2 * 1 + 3 * 4, 2 * 2 + 3 * 5, 2 * 3 + 3 * 6]
    assert result["a__b__wsum"].to_list() == pytest.approx(expected)


def test_transform_negative_coefficients(sample_df):
    # a - b (i.e., 1*a + (-1)*b)
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        coefficients=[[1.0, -1.0]],
    )
    result = transformer.fit_transform(sample_df)

    expected = [1 - 4, 2 - 5, 3 - 6]
    assert result["a__b__wsum"].to_list() == pytest.approx(expected)


def test_transform_zero_coefficient(sample_df):
    # 0*a + 1*b = b
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        coefficients=[[0.0, 1.0]],
    )
    result = transformer.fit_transform(sample_df)

    assert result["a__b__wsum"].to_list() == pytest.approx([4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# Bias term
# ---------------------------------------------------------------------------


def test_transform_with_positive_bias(sample_df):
    # a + b + 10
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        biases=[10.0],
    )
    result = transformer.fit_transform(sample_df)

    expected = [5.0 + 10, 7.0 + 10, 9.0 + 10]
    assert result["a__b__wsum"].to_list() == pytest.approx(expected)


def test_transform_with_negative_bias(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a"]],
        biases=[-5.0],
    )
    result = transformer.fit_transform(sample_df)

    assert result["a__wsum"].to_list() == pytest.approx([-4.0, -3.0, -2.0])


def test_transform_with_zero_bias(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        biases=[0.0],
    )
    result = transformer.fit_transform(sample_df)

    assert result["a__b__wsum"].to_list() == pytest.approx([5.0, 7.0, 9.0])


def test_transform_coefficients_and_bias_combined(sample_df):
    # 2*a - 1*b + 3
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        coefficients=[[2.0, -1.0]],
        biases=[3.0],
    )
    result = transformer.fit_transform(sample_df)

    expected = [2 * 1 - 4 + 3, 2 * 2 - 5 + 3, 2 * 3 - 6 + 3]
    assert result["a__b__wsum"].to_list() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Multiple output features
# ---------------------------------------------------------------------------


def test_transform_multiple_groups(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"], ["b", "c"]],
    )
    result = transformer.fit_transform(sample_df)

    assert "a__b__wsum" in result.columns
    assert "b__c__wsum" in result.columns
    assert result["a__b__wsum"].to_list() == pytest.approx([5.0, 7.0, 9.0])
    assert result["b__c__wsum"].to_list() == pytest.approx([14.0, 25.0, 36.0])


def test_transform_multiple_groups_shape(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a"], ["b"], ["c"]],
    )
    result = transformer.fit_transform(sample_df)

    # 4 original + 3 new = 7
    assert result.shape == (3, 7)


def test_transform_multiple_groups_with_biases(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"], ["b", "c"]],
        coefficients=[[1.0, 2.0], [3.0, 4.0]],
        biases=[1.0, -1.0],
    )
    result = transformer.fit_transform(sample_df)

    expected_ab = [1 + 2 * 4 + 1, 2 + 2 * 5 + 1, 3 + 2 * 6 + 1]
    expected_bc = [3 * 4 + 4 * 10 - 1, 3 * 5 + 4 * 20 - 1, 3 * 6 + 4 * 30 - 1]
    assert result["a__b__wsum"].to_list() == pytest.approx(expected_ab)
    assert result["b__c__wsum"].to_list() == pytest.approx(expected_bc)


# ---------------------------------------------------------------------------
# Custom column names
# ---------------------------------------------------------------------------


def test_transform_custom_column_names(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        new_column_names=["score"],
    )
    result = transformer.fit_transform(sample_df)

    assert "score" in result.columns
    assert "a__b__wsum" not in result.columns


def test_fit_does_not_overwrite_custom_column_names(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a"]],
        new_column_names=["my_feature"],
    )
    transformer.fit(sample_df)

    assert transformer.new_column_names == ["my_feature"]
    result = transformer.transform(sample_df)
    assert "my_feature" in result.columns


def test_transform_multiple_custom_names(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"], ["b", "c"]],
        new_column_names=["feat_1", "feat_2"],
    )
    result = transformer.fit_transform(sample_df)

    assert "feat_1" in result.columns
    assert "feat_2" in result.columns


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------


def test_transform_drop_columns(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(sample_df)

    assert "a" not in result.columns
    assert "b" not in result.columns
    assert "a__b__wsum" in result.columns
    assert "c" in result.columns
    assert "other" in result.columns


def test_transform_drop_columns_multiple_groups():
    X = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0], "extra": [9.0]})
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"], ["c", "d"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    for col in ("a", "b", "c", "d"):
        assert col not in result.columns
    assert "extra" in result.columns
    assert "a__b__wsum" in result.columns
    assert "c__d__wsum" in result.columns


def test_transform_drop_columns_shared_column():
    # Column used in two groups must not cause a double-drop error
    X = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"], ["b", "c"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    for col in ("a", "b", "c"):
        assert col not in result.columns


def test_transform_no_drop_columns_by_default(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a", "b"]])
    result = transformer.fit_transform(sample_df)

    assert "a" in result.columns
    assert "b" in result.columns


# ---------------------------------------------------------------------------
# fit() internals
# ---------------------------------------------------------------------------


def test_fit_returns_self(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a", "b"]])
    assert transformer.fit(sample_df) is transformer


def test_fit_sets_default_coefficients(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a", "b"], ["c"]])
    transformer.fit(sample_df)

    assert transformer.coefficients == [[1.0, 1.0], [1.0]]


def test_fit_preserves_provided_coefficients(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        coefficients=[[3.0, 0.5]],
    )
    transformer.fit(sample_df)

    assert transformer.coefficients == [[3.0, 0.5]]


def test_fit_sets_default_biases(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a"], ["b"]])
    transformer.fit(sample_df)

    assert transformer.biases == [0.0, 0.0]


def test_fit_preserves_provided_biases(sample_df):
    transformer = WeightedSumFeatures(
        column_groups=[["a"]],
        biases=[7.0],
    )
    transformer.fit(sample_df)

    assert transformer.biases == [7.0]


def test_fit_sets_default_new_column_names(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a", "b"]])
    transformer.fit(sample_df)

    assert transformer.new_column_names == ["a__b__wsum"]


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_get_params():
    transformer = WeightedSumFeatures(
        column_groups=[["a", "b"]],
        coefficients=[[2.0, 3.0]],
        biases=[1.0],
    )
    params = transformer.get_params()

    assert params["column_groups"] == [["a", "b"]]
    assert params["coefficients"] == [[2.0, 3.0]]
    assert params["biases"] == [1.0]
    assert params["new_column_names"] is None
    assert params["drop_columns"] is False


def test_set_params(sample_df):
    transformer = WeightedSumFeatures(column_groups=[["a"]])
    transformer.set_params(drop_columns=True)

    assert transformer.drop_columns is True


def test_fit_transform_equivalent_to_fit_then_transform(sample_df):
    t1 = WeightedSumFeatures(column_groups=[["a", "b"]], coefficients=[[2.0, -1.0]])
    t2 = WeightedSumFeatures(column_groups=[["a", "b"]], coefficients=[[2.0, -1.0]])
    t1.fit(sample_df)

    assert_frame_equal(t1.transform(sample_df), t2.fit_transform(sample_df))


# ---------------------------------------------------------------------------
# Null behavior
# ---------------------------------------------------------------------------


def test_transform_null_treated_as_zero_by_sum_horizontal():
    # pl.sum_horizontal ignores nulls (treats them as 0)
    X = pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    transformer = WeightedSumFeatures(column_groups=[["a", "b"]])
    result = transformer.fit_transform(X)

    # Row 0: 1+4=5, Row 1: null+5=5 (null ignored), Row 2: 3+null=3 (null ignored)
    assert result["a__b__wsum"].to_list() == pytest.approx([5.0, 5.0, 3.0])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_empty_group():
    with pytest.raises(ValidationError, match="at least one column"):
        WeightedSumFeatures(column_groups=[[]])


def test_validation_coefficients_outer_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        WeightedSumFeatures(
            column_groups=[["a"], ["b"]],
            coefficients=[[1.0]],
        )


def test_validation_coefficients_inner_length_mismatch():
    with pytest.raises(ValidationError, match="has 3 entries"):
        WeightedSumFeatures(
            column_groups=[["a", "b"]],
            coefficients=[[1.0, 2.0, 3.0]],
        )


def test_validation_biases_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        WeightedSumFeatures(
            column_groups=[["a"], ["b"]],
            biases=[0.0],
        )


def test_validation_new_column_names_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        WeightedSumFeatures(
            column_groups=[["a"]],
            new_column_names=["n1", "n2"],
        )


def test_validation_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        WeightedSumFeatures(column_groups=[["a"]], unknown=True)
