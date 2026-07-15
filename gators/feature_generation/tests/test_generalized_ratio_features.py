import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation import GeneralizedRatioFeatures


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "f1": [5.0, 20.0, 0.0],
            "f2": [500.0, 200.0, 0.0],
            "f3": [60.0, 100.0, 10.0],
            "f4": [6000.0, 4000.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# Basic transform — single ratio, equal weights
# ---------------------------------------------------------------------------


def test_transform_single_ratio_single_columns(sample_df):
    # (f1) / (f3 + 1)
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
    )
    result = transformer.fit_transform(sample_df)

    expected_vals = [5.0 / 61.0, 20.0 / 101.0, 0.0 / 11.0]
    assert result["f1__gratio__f3"].to_list() == pytest.approx(expected_vals)


def test_transform_single_ratio_multiple_columns(sample_df):
    # (f1 + f2) / (f3 + f4 + 1)
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3", "f4"]],
    )
    result = transformer.fit_transform(sample_df)

    expected_vals = [
        (5.0 + 500.0) / (60.0 + 6000.0 + 1.0),
        (20.0 + 200.0) / (100.0 + 4000.0 + 1.0),
        (0.0 + 0.0) / (10.0 + 0.0 + 1.0),
    ]
    assert result["f1__f2__gratio__f3__f4"].to_list() == pytest.approx(expected_vals)


def test_output_column_name_auto_generated(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3", "f4"]],
    )
    result = transformer.fit_transform(sample_df)

    assert "f1__f2__gratio__f3__f4" in result.columns


# ---------------------------------------------------------------------------
# Custom coefficients
# ---------------------------------------------------------------------------


def test_transform_custom_numerator_coefficients(sample_df):
    # numerator = 1*f1 + 2*f2; denominator = 1*f3 + 1
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3"]],
        numerator_coefficients=[[1.0, 2.0]],
    )
    result = transformer.fit_transform(sample_df)

    expected_vals = [
        (1.0 * 5.0 + 2.0 * 500.0) / (60.0 + 1.0),
        (1.0 * 20.0 + 2.0 * 200.0) / (100.0 + 1.0),
        (1.0 * 0.0 + 2.0 * 0.0) / (10.0 + 1.0),
    ]
    assert result["f1__f2__gratio__f3"].to_list() == pytest.approx(expected_vals)


def test_transform_custom_denominator_coefficients(sample_df):
    # numerator = f1; denominator = 0.5*f3 + 1
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        denominator_coefficients=[[0.5]],
    )
    result = transformer.fit_transform(sample_df)

    expected_vals = [
        5.0 / (0.5 * 60.0 + 1.0),
        20.0 / (0.5 * 100.0 + 1.0),
        0.0 / (0.5 * 10.0 + 1.0),
    ]
    assert result["f1__gratio__f3"].to_list() == pytest.approx(expected_vals)


def test_transform_both_custom_coefficients_asymmetric_groups(sample_df):
    # numerator = 1*f1 + 2*f2; denominator = 0.5*f3 + 0.1*f4 + 1
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3", "f4"]],
        numerator_coefficients=[[1.0, 2.0]],
        denominator_coefficients=[[0.5, 0.1]],
    )
    result = transformer.fit_transform(sample_df)

    expected_vals = [
        (1.0 * 5.0 + 2.0 * 500.0) / (0.5 * 60.0 + 0.1 * 6000.0 + 1.0),
        (1.0 * 20.0 + 2.0 * 200.0) / (0.5 * 100.0 + 0.1 * 4000.0 + 1.0),
        (1.0 * 0.0 + 2.0 * 0.0) / (0.5 * 10.0 + 0.1 * 0.0 + 1.0),
    ]
    assert result["f1__f2__gratio__f3__f4"].to_list() == pytest.approx(expected_vals)


def test_transform_different_num_denom_group_sizes():
    # 2 numerator cols, 1 denominator col
    X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [10.0, 20.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["a", "b"]],
        denominator_columns=[["c"]],
    )
    result = transformer.fit_transform(X)

    # (1+3)/(10+1)=4/11, (2+4)/(20+1)=6/21
    expected_vals = [4.0 / 11.0, 6.0 / 21.0]
    assert result["a__b__gratio__c"].to_list() == pytest.approx(expected_vals)


# ---------------------------------------------------------------------------
# Multiple ratios
# ---------------------------------------------------------------------------


def test_transform_multiple_ratios(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"], ["f2"]],
        denominator_columns=[["f3"], ["f4"]],
    )
    result = transformer.fit_transform(sample_df)

    assert "f1__gratio__f3" in result.columns
    assert "f2__gratio__f4" in result.columns

    cnt_vals = [5.0 / 61.0, 20.0 / 101.0, 0.0 / 11.0]
    amt_vals = [500.0 / 6001.0, 200.0 / 4001.0, 0.0 / 1.0]
    assert result["f1__gratio__f3"].to_list() == pytest.approx(cnt_vals)
    assert result["f2__gratio__f4"].to_list() == pytest.approx(amt_vals)


def test_transform_multiple_ratios_all_columns_appended(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"], ["f2"]],
        denominator_columns=[["f3"], ["f4"]],
    )
    result = transformer.fit_transform(sample_df)

    # original 4 columns + 2 new = 6
    assert result.shape == (3, 6)


# ---------------------------------------------------------------------------
# Epsilon
# ---------------------------------------------------------------------------


def test_transform_epsilon_prevents_division_by_zero():
    # Denominator column is zero; epsilon=1 → denom_sum+1 = 1 → ratio = numerator
    X = pl.DataFrame({"num": [5.0, 0.0], "denom": [0.0, 0.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["num"]],
        denominator_columns=[["denom"]],
        epsilon=1.0,
    )
    result = transformer.fit_transform(X)

    assert result["num__gratio__denom"].to_list() == pytest.approx([5.0, 0.0])


def test_transform_custom_epsilon():
    X = pl.DataFrame({"num": [1.0], "denom": [9.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["num"]],
        denominator_columns=[["denom"]],
        epsilon=1.0,
    )
    result = transformer.fit_transform(X)

    # 1 / (9 + 1) = 0.1
    assert result["num__gratio__denom"][0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Null propagation
# ---------------------------------------------------------------------------


def test_transform_null_in_numerator_treated_as_zero():
    # pl.sum_horizontal ignores nulls (treats them as 0), so a null numerator
    # column produces 0 / (denom + epsilon), not null.
    X = pl.DataFrame({"num": [10.0, None, 30.0], "denom": [5.0, 5.0, 5.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["num"]],
        denominator_columns=[["denom"]],
    )
    result = transformer.fit_transform(X)

    vals = result["num__gratio__denom"].to_list()
    assert vals[0] == pytest.approx(10.0 / 6.0)
    assert vals[1] == pytest.approx(0.0)  # null → 0 via sum_horizontal
    assert vals[2] == pytest.approx(30.0 / 6.0)


def test_transform_null_in_denominator_treated_as_zero():
    # sum_horizontal ignores nulls (treats as 0), so denom_sum = 0 → +epsilon
    X = pl.DataFrame({"num": [10.0], "denom": [None]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["num"]],
        denominator_columns=[["denom"]],
        epsilon=1.0,
    )
    result = transformer.fit_transform(X)

    # denom null → sum_horizontal = 0 → +1 = 1 → 10/1 = 10.0
    assert result["num__gratio__denom"][0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Custom column names
# ---------------------------------------------------------------------------


def test_transform_custom_column_names(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3", "f4"]],
        new_column_names=["composite_ratio"],
    )
    result = transformer.fit_transform(sample_df)

    assert "composite_ratio" in result.columns
    assert "f1__f2__gratio__f3__f4" not in result.columns


def test_fit_does_not_overwrite_custom_column_names(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        new_column_names=["my_ratio"],
    )
    transformer.fit(sample_df)

    assert transformer.new_column_names == ["my_ratio"]
    result = transformer.transform(sample_df)
    assert "my_ratio" in result.columns


# ---------------------------------------------------------------------------
# drop_columns
# ---------------------------------------------------------------------------


def test_transform_drop_columns(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(sample_df)

    assert "f1" not in result.columns
    assert "f3" not in result.columns
    assert "f1__gratio__f3" in result.columns
    # Unrelated columns kept
    assert "f2" in result.columns
    assert "f4" in result.columns


def test_transform_drop_columns_multi_group():
    # Columns used across both numerator and denominator groups must all be dropped
    X = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0], "extra": [9.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["a", "b"], ["c"]],
        denominator_columns=[["c", "d"], ["a"]],
        drop_columns=True,
    )
    result = transformer.fit_transform(X)

    for col in ("a", "b", "c", "d"):
        assert col not in result.columns
    assert "extra" in result.columns


def test_transform_no_drop_columns_by_default(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
    )
    result = transformer.fit_transform(sample_df)

    assert "f1" in result.columns
    assert "f3" in result.columns


# ---------------------------------------------------------------------------
# Integer input → Float64 output
# ---------------------------------------------------------------------------


def test_transform_integer_columns_cast_to_float64():
    X = pl.DataFrame({"num": [10, 20], "denom": [5, 10]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["num"]],
        denominator_columns=[["denom"]],
    )
    result = transformer.fit_transform(X)

    assert result["num__gratio__denom"].dtype == pl.Float64


# ---------------------------------------------------------------------------
# fit() internals
# ---------------------------------------------------------------------------


def test_fit_returns_self(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
    )
    assert transformer.fit(sample_df) is transformer


def test_fit_sets_default_numerator_coefficients(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1", "f2"]],
        denominator_columns=[["f3"]],
    )
    transformer.fit(sample_df)

    assert transformer.numerator_coefficients == [[1.0, 1.0]]


def test_fit_sets_default_denominator_coefficients(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3", "f4"]],
    )
    transformer.fit(sample_df)

    assert transformer.denominator_coefficients == [[1.0, 1.0]]


def test_fit_preserves_provided_coefficients(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        numerator_coefficients=[[3.0]],
        denominator_coefficients=[[0.5]],
    )
    transformer.fit(sample_df)

    assert transformer.numerator_coefficients == [[3.0]]
    assert transformer.denominator_coefficients == [[0.5]]


def test_fit_sets_default_new_column_names(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
    )
    transformer.fit(sample_df)

    assert transformer.new_column_names == ["f1__gratio__f3"]


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_get_params():
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        epsilon=2.0,
    )
    params = transformer.get_params()

    assert params["numerator_columns"] == [["f1"]]
    assert params["denominator_columns"] == [["f3"]]
    assert params["epsilon"] == 2.0
    assert params["numerator_coefficients"] is None
    assert params["denominator_coefficients"] is None
    assert params["new_column_names"] is None
    assert params["drop_columns"] is False


def test_set_params():
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["a"]],
        denominator_columns=[["b"]],
    )
    transformer.set_params(drop_columns=True, epsilon=0.1)

    assert transformer.drop_columns is True
    assert transformer.epsilon == pytest.approx(0.1)


def test_fit_transform_equivalent_to_fit_then_transform(sample_df):
    t1 = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
    )
    t2 = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
    )
    t1.fit(sample_df)
    result_pipe = t1.transform(sample_df)
    result_combined = t2.fit_transform(sample_df)

    assert_frame_equal(result_pipe, result_combined)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_num_denom_outer_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"], ["b"]],
            denominator_columns=[["x"]],
        )


def test_validation_empty_denominator_group():
    with pytest.raises(ValidationError, match="at least one column"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"]],
            denominator_columns=[[]],
        )


def test_validation_numerator_coefficients_outer_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"], ["b"]],
            denominator_columns=[["x"], ["y"]],
            numerator_coefficients=[[1.0]],
        )


def test_validation_numerator_coefficients_inner_length_mismatch():
    with pytest.raises(ValidationError, match="has 3 entries"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a", "b"]],
            denominator_columns=[["x"]],
            numerator_coefficients=[[1.0, 2.0, 3.0]],
        )


def test_validation_denominator_coefficients_outer_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"]],
            denominator_columns=[["x"], ["y"]],
            denominator_coefficients=[[1.0]],
        )


def test_validation_denominator_coefficients_inner_length_mismatch():
    with pytest.raises(ValidationError, match="has 1 entries"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"]],
            denominator_columns=[["x", "y"]],
            denominator_coefficients=[[1.0]],
        )


def test_validation_new_column_names_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"]],
            denominator_columns=[["x"]],
            new_column_names=["n1", "n2"],
        )


def test_validation_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"]],
            denominator_columns=[["x"]],
            unknown_param=True,
        )


def test_validation_epsilon_must_be_positive():
    with pytest.raises(ValidationError):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"]],
            denominator_columns=[["x"]],
            epsilon=0.0,
        )


def test_validation_numerator_biases_length_mismatch():
    with pytest.raises(ValidationError, match="must match length"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"], ["b"]],
            denominator_columns=[["x"], ["y"]],
            numerator_biases=[1.0],
        )


def test_validation_new_column_names_duplicates():
    with pytest.raises(ValidationError, match="duplicate names"):
        GeneralizedRatioFeatures(
            numerator_columns=[["a"], ["b"]],
            denominator_columns=[["x"], ["y"]],
            new_column_names=["ratio", "ratio"],
        )


def test_fit_raises_on_duplicate_auto_generated_names():
    # Two identical (numerator, denominator) pairs produce the same default name
    X = pl.DataFrame({"a": [1.0], "b": [2.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["a"], ["a"]],
        denominator_columns=[["b"], ["b"]],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        transformer.fit(X)


def test_fit_raises_on_output_name_clashing_with_existing_column(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        new_column_names=["f2"],  # f2 already exists in sample_df
    )
    with pytest.raises(ValueError, match="already exist"):
        transformer.fit(sample_df)


# ---------------------------------------------------------------------------
# Constant numerator (empty numerator_columns inner list)
# ---------------------------------------------------------------------------


def test_transform_constant_numerator():
    # 1 / (f3 + f4 + epsilon)
    X = pl.DataFrame({"f3": [9.0, 4.0], "f4": [0.0, 5.0]})
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[[]],
        denominator_columns=[["f3", "f4"]],
        numerator_biases=[1.0],
    )
    result = transformer.fit_transform(X)

    # (9+0+1)=10, (4+5+1)=10
    assert result["const__gratio__f3__f4"].to_list() == pytest.approx([0.1, 0.1])


def test_fit_sets_default_numerator_biases(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"], ["f2"]],
        denominator_columns=[["f3"], ["f4"]],
    )
    transformer.fit(sample_df)

    assert transformer.numerator_biases == [0.0, 0.0]


def test_fit_preserves_provided_numerator_biases(sample_df):
    transformer = GeneralizedRatioFeatures(
        numerator_columns=[["f1"]],
        denominator_columns=[["f3"]],
        numerator_biases=[5.0],
    )
    transformer.fit(sample_df)

    assert transformer.numerator_biases == [5.0]
