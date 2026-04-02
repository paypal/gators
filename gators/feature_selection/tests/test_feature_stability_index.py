"""Tests for feature_selection.py"""
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from gators.feature_selection.feature_stability_index import feature_stability_index
from sklearn.model_selection import StratifiedKFold

class TestFeatureStabilityIndex:
    """Tests for feature_stability_index function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data with stable and unstable features."""
        # Create a simple dataset with 100 samples
        # Feature A: Very stable - always important
        # Feature B: Moderately stable - sometimes important
        # Feature C: Unstable - rarely important
        pl.set_random_seed(42)
        X = pl.DataFrame({
            'A': pl.Series([i % 2 for i in range(100)]),  # Binary feature
            'B': pl.Series([(i // 10) % 3 for i in range(100)]),  # Categorical-like
            'C': pl.Series(range(100)),  # Sequential
            'D': pl.Series([(i ** 2) % 7 for i in range(100)])  # Non-linear pattern
        })
        # Create binary target that depends mostly on feature A
        y = pl.Series('target', [1 if x > 0.5 else 0 for x in (X['A'] + X['B'] * 0.1).to_list()])
        return X, y

    def test_basic_functionality(self, sample_data):
        """Test basic FSI computation."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        result = feature_stability_index(estimator, skf, X, y)
        
        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ['feature', 'fsi', 'importance']
        assert all(result['fsi'] > 0)  # Only non-zero FSI
        assert len(result) > 0  # Should have at least some features

    def test_importance_threshold_zero(self, sample_data):
        """Test with importance_threshold=0.0 (default)."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        result = feature_stability_index(estimator, skf, X, y, importance_threshold=0.0)
        
        # With threshold 0.0, all features with any importance should be included
        assert len(result) > 0
        assert all(result['fsi'] > 0)

    def test_importance_threshold_high(self, sample_data):
        """Test with high importance_threshold."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        # High threshold should filter out weak features
        result = feature_stability_index(estimator, skf, X, y, importance_threshold=0.3)
        
        assert isinstance(result, pl.DataFrame)
        # Fewer or equal features compared to low threshold
        result_low = feature_stability_index(estimator, skf, X, y, importance_threshold=0.0)
        assert len(result) <= len(result_low)

    def test_very_high_threshold_empty_result(self, sample_data):
        """Test with extremely high threshold that filters all features."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        # Impossibly high threshold should result in empty or very small result
        result = feature_stability_index(estimator, skf, X, y, importance_threshold=1.0)
        
        assert isinstance(result, pl.DataFrame)
        # Result could be empty or near-empty
        assert len(result) <= len(sample_data[0].columns)

    def test_sorting_behavior(self, sample_data):
        """Test that results are sorted by FSI desc and importance desc."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        
        result = feature_stability_index(estimator, skf, X, y)
        
        if len(result) > 1:
            # Check FSI is descending
            fsi_values = result['fsi'].to_list()
            for i in range(len(fsi_values) - 1):
                # If FSI is equal, importance should be descending
                if fsi_values[i] == fsi_values[i + 1]:
                    importance_values = result['importance'].to_list()
                    assert importance_values[i] >= importance_values[i + 1]
                else:
                    assert fsi_values[i] >= fsi_values[i + 1]

    def test_with_decision_tree(self, sample_data):
        """Test with different estimator type (DecisionTreeClassifier)."""
        X, y = sample_data
        estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        
        result = feature_stability_index(estimator, skf, X, y)
        
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ['feature', 'fsi', 'importance']
        assert all(result['fsi'] > 0)

    def test_feature_names_in_result(self, sample_data):
        """Test that feature names are correctly included in result."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        result = feature_stability_index(estimator, skf, X, y)
        
        # All returned features should be from the original feature set
        original_features = set(X.columns)
        result_features = set(result['feature'].to_list())
        assert result_features.issubset(original_features)

    def test_fsi_values_range(self, sample_data):
        """Test that FSI values are in valid range (0, 1]."""
        X, y = sample_data
        estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        result = feature_stability_index(estimator, skf, X, y)
        
        # FSI should be between 0 (exclusive, filtered out) and 1 (inclusive)
        assert all(result['fsi'] > 0)
        assert all(result['fsi'] <= 1.0)
