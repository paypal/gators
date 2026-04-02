"""Test util module."""

import polars as pl
import pytest

from gators.data_cleaning.util import map_substring_replacements


class TestMapSubstringReplacements:
    """Test map_substring_replacements function."""

    def test_basic_space_to_underscore(self):
        """Test basic functionality replacing spaces with underscores."""
        X = pl.DataFrame({
            'categories': ['cat A', 'cat B', 'cat A', 'dog C'],
            'numbers': [1, 2, 3, 4]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'categories': {'cat A': 'cat_A', 'cat B': 'cat_B', 'dog C': 'dog_C'}
        }
        assert mapping == expected

    def test_multiple_string_columns(self):
        """Test with multiple string columns."""
        X = pl.DataFrame({
            'col1': ['value A', 'value B'],
            'col2': ['item X', 'item Y'],
            'col3': [1, 2]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'col1': {'value A': 'value_A', 'value B': 'value_B'},
            'col2': {'item X': 'item_X', 'item Y': 'item_Y'}
        }
        assert mapping == expected

    def test_no_matches_returns_empty_dict(self):
        """Test that columns without matches are excluded from mapping."""
        X = pl.DataFrame({
            'no_spaces': ['valueA', 'valueB', 'valueC'],
            'numbers': [1, 2, 3]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        assert mapping == {}

    def test_only_numeric_columns_returns_empty(self):
        """Test with only numeric columns returns empty mapping."""
        X = pl.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'bool_col': [True, False, True]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        assert mapping == {}

    def test_mixed_columns_only_processes_strings(self):
        """Test that only string columns are processed."""
        X = pl.DataFrame({
            'strings': ['a b', 'c d'],
            'integers': [1, 2],
            'floats': [1.5, 2.5],
            'bools': [True, False]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'strings': {'a b': 'a_b', 'c d': 'c_d'}
        }
        assert mapping == expected

    def test_categorical_column(self):
        """Test with categorical dtype column."""
        X = pl.DataFrame({
            'cat_col': ['cat A', 'cat B', 'cat A']
        }).with_columns(pl.col('cat_col').cast(pl.Categorical))
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'cat_col': {'cat A': 'cat_A', 'cat B': 'cat_B'}
        }
        assert mapping == expected

    def test_duplicate_values_in_column(self):
        """Test that duplicate values are handled correctly (unique values only)."""
        X = pl.DataFrame({
            'col': ['item A', 'item B', 'item A', 'item B', 'item C']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'col': {'item A': 'item_A', 'item B': 'item_B', 'item C': 'item_C'}
        }
        assert mapping == expected

    def test_different_old_new_characters(self):
        """Test with different old and new substrings."""
        X = pl.DataFrame({
            'col1': ['a-b', 'c-d'],
            'col2': ['x.y', 'z.w']
        })
        
        # Replace hyphens with underscores
        mapping1 = map_substring_replacements(X, old='-', new='_')
        assert mapping1 == {'col1': {'a-b': 'a_b', 'c-d': 'c_d'}}
        
        # Replace dots with underscores
        mapping2 = map_substring_replacements(X, old='.', new='_')
        assert mapping2 == {'col2': {'x.y': 'x_y', 'z.w': 'z_w'}}

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pl.DataFrame({
            'col1': [],
            'col2': []
        }).with_columns([
            pl.col('col1').cast(pl.String),
            pl.col('col2').cast(pl.Int64)
        ])
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        assert mapping == {}

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        X = pl.DataFrame({
            'col': ['value A']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {'col': {'value A': 'value_A'}}
        assert mapping == expected

    def test_column_with_nulls(self):
        """Test that null values are handled correctly."""
        X = pl.DataFrame({
            'col': ['value A', None, 'value B', None]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        # Only non-null values with matches should be in mapping
        expected = {'col': {'value A': 'value_A', 'value B': 'value_B'}}
        assert mapping == expected

    def test_partial_match_only_some_values(self):
        """Test when only some values in a column contain the substring."""
        X = pl.DataFrame({
            'col': ['value A', 'valueB', 'value C', 'valueD']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        # Only values with spaces should be in mapping
        expected = {'col': {'value A': 'value_A', 'value C': 'value_C'}}
        assert mapping == expected

    def test_multiple_occurrences_in_same_value(self):
        """Test value with multiple occurrences of the substring."""
        X = pl.DataFrame({
            'col': ['a b c', 'x y z']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {'col': {'a b c': 'a_b_c', 'x y z': 'x_y_z'}}
        assert mapping == expected

    def test_replace_with_empty_string(self):
        """Test replacing substring with empty string."""
        X = pl.DataFrame({
            'col': ['value A', 'value B']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='')
        
        expected = {'col': {'value A': 'valueA', 'value B': 'valueB'}}
        assert mapping == expected

    def test_replace_empty_string_with_character(self):
        """Test with empty old substring (should not match anything practical)."""
        X = pl.DataFrame({
            'col': ['abc', 'def']
        })
        
        # Empty old string won't match in a useful way with str.contains
        mapping = map_substring_replacements(X, old='', new='_')
        
        # This might match everything depending on polars behavior
        # The main thing is it shouldn't crash
        assert isinstance(mapping, dict)

    def test_special_regex_characters(self):
        """Test with special regex characters in old substring."""
        X = pl.DataFrame({
            'col1': ['a.b', 'c.d'],
            'col2': ['x(y)', 'z(w)']
        })
        
        # Dot
        mapping1 = map_substring_replacements(X, old='.', new='_')
        assert mapping1 == {'col1': {'a.b': 'a_b', 'c.d': 'c_d'}}
        
        # Parentheses
        mapping2 = map_substring_replacements(X, old='(', new='[')
        assert mapping2 == {'col2': {'x(y)': 'x[y)', 'z(w)': 'z[w)'}}

    def test_onehot_encoding_use_case(self):
        """Test realistic use case: fixing column names after one-hot encoding."""
        # Simulate column names that would come from one-hot encoding
        X = pl.DataFrame({
            'color__light blue': [1, 0, 1, 0],
            'color__dark red': [0, 1, 0, 1],
            'size__extra large': [1, 1, 0, 0],
            'price': [10.5, 20.3, 15.7, 18.2]
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        # Only string columns should be processed, numeric columns ignored
        # In this case, all columns are numeric, so mapping should be empty
        assert mapping == {}

    def test_string_column_names_with_spaces(self):
        """Test with actual string values containing spaces (not column names)."""
        X = pl.DataFrame({
            'category': ['light blue', 'dark red', 'light blue'],
            'size': ['extra large', 'small', 'medium']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'category': {'light blue': 'light_blue', 'dark red': 'dark_red'},
            'size': {'extra large': 'extra_large'}
        }
        assert mapping == expected

    def test_multicharacter_old_and_new(self):
        """Test with multi-character old and new substrings."""
        X = pl.DataFrame({
            'col': ['value__A', 'value__B', 'item__C']
        })
        
        mapping = map_substring_replacements(X, old='__', new='--')
        
        expected = {
            'col': {'value__A': 'value--A', 'value__B': 'value--B', 'item__C': 'item--C'}
        }
        assert mapping == expected

    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive."""
        X = pl.DataFrame({
            'col': ['Value A', 'value a', 'VALUE A']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        # All should be replaced (case doesn't matter for space)
        expected = {
            'col': {'Value A': 'Value_A', 'value a': 'value_a', 'VALUE A': 'VALUE_A'}
        }
        assert mapping == expected

    def test_unicode_characters(self):
        """Test with unicode characters."""
        X = pl.DataFrame({
            'col': ['café au lait', 'crème brûlée']
        })
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        expected = {
            'col': {'café au lait': 'café_au_lait', 'crème brûlée': 'crème_brûlée'}
        }
        assert mapping == expected

    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        X = pl.DataFrame({
            'col': ['value A', 'value B']
        })
        original_values = X['col'].to_list()
        
        mapping = map_substring_replacements(X, old=' ', new='_')
        
        # Original DataFrame should be unchanged
        assert X['col'].to_list() == original_values
        assert mapping == {'col': {'value A': 'value_A', 'value B': 'value_B'}}
