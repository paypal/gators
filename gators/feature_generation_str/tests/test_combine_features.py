import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from gators.feature_generation_str import CombineFeatures


def test_transform_basic_single_combination():
    """Test basic transformation with single column combination."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "A", "B", "B", "A"],
            "cat2": ["X", "Y", "X", "Y", "X"],
            "amount": [100, 200, 150, 300, 250],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"]])
    result = transformer.fit_transform(X)

    assert "cat1__cat2" in result.columns
    assert result["cat1__cat2"][0] == "A_X"
    assert result["cat1__cat2"][1] == "A_Y"
    assert result["cat1__cat2"][2] == "B_X"


def test_transform_multiple_combinations():
    """Test with multiple column combinations."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "A", "B", "B"],
            "cat2": ["X", "Y", "X", "Y"],
            "addr1": ["US", "US", "UK", "UK"],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"], ["cat1", "addr1"]])
    result = transformer.fit_transform(X)

    assert "cat1__cat2" in result.columns
    assert "cat1__addr1" in result.columns
    assert result["cat1__cat2"][0] == "A_X"
    assert result["cat1__addr1"][0] == "A_US"


def test_transform_three_columns():
    """Test combining three columns."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "A", "B"],
            "cat2": ["X", "Y", "X"],
            "card3": ["1", "2", "1"],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2", "card3"]])
    result = transformer.fit_transform(X)

    assert "cat1__cat2__card3" in result.columns
    assert result["cat1__cat2__card3"][0] == "A_X_1"
    assert result["cat1__cat2__card3"][1] == "A_Y_2"


def test_transform_with_custom_separator():
    """Test using custom separator."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B"],
            "cat2": ["X", "Y"],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"]], separator="|")
    result = transformer.fit_transform(X)

    assert result["cat1__cat2"][0] == "A|X"
    assert result["cat1__cat2"][1] == "B|Y"


def test_transform_with_custom_column_names():
    """Test using custom column names."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B"],
            "cat2": ["X", "Y"],
            "addr1": ["US", "UK"],
        }
    )

    transformer = CombineFeatures(
        column_groups=[["cat1", "cat2"], ["cat1", "addr1"]],
        new_column_names=["card_combo", "card_location"],
    )
    result = transformer.fit_transform(X)

    assert "card_combo" in result.columns
    assert "card_location" in result.columns
    assert "cat1__cat2" not in result.columns


def test_transform_with_drop_columns():
    """Test dropping original columns."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B"],
            "cat2": ["X", "Y"],
            "amount": [100, 200],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"]], drop_columns=True)
    result = transformer.fit_transform(X)

    assert "cat1" not in result.columns
    assert "cat2" not in result.columns
    assert "cat1__cat2" in result.columns
    assert "amount" in result.columns


def test_transform_with_null_values():
    """Test handling of null values."""
    X = pl.DataFrame(
        {
            "cat1": ["A", None, "B"],
            "cat2": ["X", "Y", None],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"]])
    result = transformer.fit_transform(X)

    assert result["cat1__cat2"][0] == "A_X"
    assert result["cat1__cat2"][1] == "null_Y"
    assert result["cat1__cat2"][2] == "B_null"


def test_transform_numeric_columns():
    """Test combining numeric columns (cast to string)."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B"],
            "card_num": [123, 456],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "card_num"]])
    result = transformer.fit_transform(X)

    assert result["cat1__card_num"][0] == "A_123"
    assert result["cat1__card_num"][1] == "B_456"


def test_transform_mixed_types():
    """Test combining columns with mixed types."""
    X = pl.DataFrame(
        {
            "card": ["A", "B"],
            "number": [1, 2],
            "date": ["2024-01-01", "2024-01-02"],
        }
    )

    transformer = CombineFeatures(column_groups=[["card", "number", "date"]])
    result = transformer.fit_transform(X)

    assert result["card__number__date"][0] == "A_1_2024-01-01"


def test_validation_mismatched_new_column_names_length():
    """Test validation error when new_column_names length doesn't match."""
    with pytest.raises(
        ValueError,
        match="Length of new_column_names .* must match length of column_groups",
    ):
        CombineFeatures(
            column_groups=[["cat1", "cat2"], ["cat1", "addr1"]],
            new_column_names=["only_one_name"],  # Should have 2 names
        )


def test_fit_return_self():
    """Test that fit returns self."""
    X = pl.DataFrame({"col1": ["A", "B"], "col2": ["X", "Y"]})

    transformer = CombineFeatures(column_groups=[["col1", "col2"]])
    result = transformer.fit(X)

    assert result is transformer


def test_column_mapping_generation():
    """Test that column mapping is correctly generated during fit."""
    X = pl.DataFrame({"col1": ["A", "B"], "col2": ["X", "Y"]})

    transformer = CombineFeatures(
        column_groups=[["col1", "col2"]], new_column_names=["custom_name"]
    )
    transformer.fit(X)

    assert len(transformer._column_mapping) == 1
    assert "col1__col2" in transformer._column_mapping
    assert transformer._column_mapping["col1__col2"] == "custom_name"


def test_empty_dataframe():
    """Test behavior with empty dataframe."""
    X = pl.DataFrame({"col1": [], "col2": []}, schema={"col1": pl.Utf8, "col2": pl.Utf8})

    transformer = CombineFeatures(column_groups=[["col1", "col2"]])
    result = transformer.fit_transform(X)

    assert result.shape[0] == 0
    assert "col1__col2" in result.columns


def test_single_row_dataframe():
    """Test behavior with single row dataframe."""
    X = pl.DataFrame(
        {
            "cat1": ["A"],
            "cat2": ["X"],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"]])
    result = transformer.fit_transform(X)

    assert result["cat1__cat2"][0] == "A_X"


def test_uid_generation_pattern():
    """Test typical UID generation use case."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "A", "B", "B", "A"],
            "cat2": ["X", "Y", "X", "X", "X"],
            "addr1": ["US", "US", "UK", "UK", "CA"],
            "email": ["gmail", "yahoo", "gmail", "outlook", "gmail"],
        }
    )

    transformer = CombineFeatures(
        column_groups=[
            ["cat1", "cat2"],
            ["cat1", "addr1"],
            ["cat1", "cat2", "addr1"],
        ],
        new_column_names=["card_uid", "card_addr_uid", "full_uid"],
    )
    result = transformer.fit_transform(X)

    assert "card_uid" in result.columns
    assert "card_addr_uid" in result.columns
    assert "full_uid" in result.columns
    assert result["full_uid"][0] == "A_X_US"


def test_duplicate_values_create_same_uid():
    """Test that duplicate combinations create the same UID."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B", "A", "B"],
            "cat2": ["X", "Y", "X", "Y"],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1", "cat2"]])
    result = transformer.fit_transform(X)

    # Rows 0 and 2 should have same UID
    assert result["cat1__cat2"][0] == result["cat1__cat2"][2]
    # Rows 1 and 3 should have same UID
    assert result["cat1__cat2"][1] == result["cat1__cat2"][3]


def test_special_characters_in_values():
    """Test handling of special characters in values."""
    X = pl.DataFrame(
        {
            "col1": ["A@B", "C-D"],
            "col2": ["X#Y", "Z$W"],
        }
    )

    transformer = CombineFeatures(column_groups=[["col1", "col2"]])
    result = transformer.fit_transform(X)

    assert result["col1__col2"][0] == "A@B_X#Y"
    assert result["col1__col2"][1] == "C-D_Z$W"


def test_long_separator():
    """Test with multi-character separator."""
    X = pl.DataFrame(
        {
            "col1": ["A", "B"],
            "col2": ["X", "Y"],
        }
    )

    transformer = CombineFeatures(column_groups=[["col1", "col2"]], separator="||")
    result = transformer.fit_transform(X)

    assert result["col1__col2"][0] == "A||X"
    assert result["col1__col2"][1] == "B||Y"


def test_empty_string_separator():
    """Test with empty string separator."""
    X = pl.DataFrame(
        {
            "col1": ["A", "B"],
            "col2": ["X", "Y"],
        }
    )

    transformer = CombineFeatures(column_groups=[["col1", "col2"]], separator="")
    result = transformer.fit_transform(X)

    assert result["col1__col2"][0] == "AX"
    assert result["col1__col2"][1] == "BY"


def test_combining_single_column():
    """Test 'combining' just one column (edge case)."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B", "C"],
            "other": [1, 2, 3],
        }
    )

    transformer = CombineFeatures(column_groups=[["cat1"]])
    result = transformer.fit_transform(X)

    assert "cat1" in result.columns
    # Should just be the string representation
    assert result["cat1"][0] == "A"


def test_many_columns_combination():
    """Test combining many columns."""
    X = pl.DataFrame(
        {
            "col1": ["A"],
            "col2": ["B"],
            "col3": ["C"],
            "col4": ["D"],
            "col5": ["E"],
        }
    )

    transformer = CombineFeatures(column_groups=[["col1", "col2", "col3", "col4", "col5"]])
    result = transformer.fit_transform(X)

    assert result["col1__col2__col3__col4__col5"][0] == "A_B_C_D_E"


def test_overlapping_column_groups_no_drop():
    """Test overlapping column groups without dropping columns."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B"],
            "cat2": ["X", "Y"],
            "card3": ["1", "2"],
        }
    )

    transformer = CombineFeatures(
        column_groups=[["cat1", "cat2"], ["cat2", "card3"]], drop_columns=False
    )
    result = transformer.fit_transform(X)

    # Original columns remain
    assert "cat1" in result.columns
    assert "cat2" in result.columns
    assert "card3" in result.columns
    # New combinations created
    assert "cat1__cat2" in result.columns
    assert "cat2__card3" in result.columns


def test_overlapping_column_groups_with_drop():
    """Test overlapping column groups with drop_columns."""
    X = pl.DataFrame(
        {
            "cat1": ["A", "B"],
            "cat2": ["X", "Y"],
            "card3": ["1", "2"],
            "amount": [100, 200],
        }
    )

    transformer = CombineFeatures(
        column_groups=[["cat1", "cat2"], ["cat2", "card3"]], drop_columns=True
    )
    result = transformer.fit_transform(X)

    # All columns in any group should be dropped
    assert "cat1" not in result.columns
    assert "cat2" not in result.columns
    assert "card3" not in result.columns
    # Amount not in any group, should remain
    assert "amount" in result.columns


def test_whitespace_in_values():
    """Test handling of whitespace in values."""
    X = pl.DataFrame(
        {
            "col1": ["A ", " B", "C"],
            "col2": ["X", "Y ", " Z"],
        }
    )

    transformer = CombineFeatures(column_groups=[["col1", "col2"]])
    result = transformer.fit_transform(X)

    # Whitespace preserved
    assert result["col1__col2"][0] == "A _X"
    assert result["col1__col2"][1] == " B_Y "


def test_boolean_columns():
    """Test combining boolean columns."""
    X = pl.DataFrame(
        {
            "flag1": [True, False, True],
            "flag2": [False, False, True],
        }
    )

    transformer = CombineFeatures(column_groups=[["flag1", "flag2"]])
    result = transformer.fit_transform(X)

    assert result["flag1__flag2"][0] == "true_false"
    assert result["flag1__flag2"][1] == "false_false"
    assert result["flag1__flag2"][2] == "true_true"


def test_all_null_column():
    """Test combining with a column that's all nulls."""
    X = pl.DataFrame(
        {
            "col1": ["A", "B", "C"],
            "col2": [None, None, None],
        }
    )

    transformer = CombineFeatures(column_groups=[["col1", "col2"]])
    result = transformer.fit_transform(X)

    assert result["col1__col2"][0] == "A_null"
    assert result["col1__col2"][1] == "B_null"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
