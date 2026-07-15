"""Tests for select_k_best_stable_features."""

import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from gators.feature_selection.select_k_best_stable_features import select_k_best_stable_features


@pytest.fixture
def clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


@pytest.fixture
def sample_data():
    X = pl.DataFrame(
        {
            "a": [i % 2 for i in range(40)],
            "b": [i % 3 for i in range(40)],
            "c": [0] * 40,
        }
    )
    y = pl.Series("target", [i % 2 for i in range(40)])
    return X, y


def test_returns_dataframe(clf, sample_data):
    X, y = sample_data
    result = select_k_best_stable_features(clf, StratifiedKFold(n_splits=2), X, y, k=2)
    assert isinstance(result, pl.DataFrame)


def test_result_columns(clf, sample_data):
    X, y = sample_data
    result = select_k_best_stable_features(clf, StratifiedKFold(n_splits=2), X, y, k=2)
    assert result.columns == ["feature", "importance"]


def test_result_sorted_descending(clf, sample_data):
    X, y = sample_data
    result = select_k_best_stable_features(clf, StratifiedKFold(n_splits=2), X, y, k=3)
    importances = result["importance"].to_list()
    assert importances == sorted(importances, reverse=True)


def test_stable_features_subset_of_all(clf, sample_data):
    X, y = sample_data
    result = select_k_best_stable_features(clf, StratifiedKFold(n_splits=2), X, y, k=2)
    assert set(result["feature"].to_list()).issubset(set(X.columns))


def test_k_equal_total_features_returns_all(clf, sample_data):
    X, y = sample_data
    result = select_k_best_stable_features(clf, KFold(n_splits=2), X, y, k=3)
    assert set(result["feature"].to_list()) == set(X.columns)


def test_importance_values_are_non_negative(clf, sample_data):
    X, y = sample_data
    result = select_k_best_stable_features(clf, StratifiedKFold(n_splits=2), X, y, k=2)
    assert all(v >= 0 for v in result["importance"].to_list())


def test_print_output(clf, sample_data, capsys):
    X, y = sample_data
    select_k_best_stable_features(clf, StratifiedKFold(n_splits=2), X, y, k=2)
    captured = capsys.readouterr()
    assert "[fsi]" in captured.out
    assert "stable features" in captured.out
