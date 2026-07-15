"""Tests for KNNImputer."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gators.imputers.knn_imputer import KNNImputer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def X_simple():
    """Single null column; B is used for distance."""
    return pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, None],
            "B": [10.0, 20.0, 30.0, 40.0, 35.0],
        }
    )


@pytest.fixture
def X_multi_null():
    """Multiple columns with nulls in different patterns."""
    return pl.DataFrame(
        {
            "A": [1.0, 2.0, None, 4.0],
            "B": [10.0, 20.0, 30.0, None],
            "C": [1.0, 2.0, 3.0, 4.0],
        }
    )


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_self(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        assert imp.fit(X_simple) is imp

    def test_fit_stores_complete_rows_only(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        # Row 4 (A is null) must NOT be in the training pool
        assert len(imp._train_df_) == 4

    def test_fit_auto_detects_null_columns(self):
        X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, 6.0]})
        imp = KNNImputer(n_neighbors=1)
        imp.fit(X)
        assert imp.subset == ["A"]

    def test_fit_detects_feature_cols(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        assert imp._feature_cols_ == ["B"]

    def test_fit_global_stats_median(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        # Median of [1, 2, 3, 4] = 2.5
        assert imp._global_stats_["A"] == pytest.approx(2.5)

    def test_fit_ignores_non_numeric_in_feature_cols(self):
        X = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, 6.0], "C": ["x", "y", "z"]})
        imp = KNNImputer(n_neighbors=1, subset=["A"])
        imp.fit(X)
        assert "C" not in imp._feature_cols_


# ---------------------------------------------------------------------------
# transform – uniform weights
# ---------------------------------------------------------------------------


class TestTransformUniform:
    def test_no_nulls_returns_unchanged(self):
        X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X)
        assert_frame_equal(imp.transform(X), X)

    def test_row_order_preserved(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        assert result["B"].to_list() == [10.0, 20.0, 30.0, 40.0, 35.0]

    def test_non_imputed_rows_unchanged(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        assert result["A"].to_list()[:4] == [1.0, 2.0, 3.0, 4.0]

    def test_imputed_value_is_mean_of_2_nearest(self, X_simple):
        # Query: B=35. Training distances: |35-10|=25, |35-20|=15, |35-30|=5, |35-40|=5
        # Top-2 nearest: rows with B=30 (A=3) and B=40 (A=4)  → mean = 3.5
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        assert result["A"][4] == pytest.approx(3.5)

    def test_imputed_value_single_neighbor(self, X_simple):
        # With n_neighbors=1, nearest to B=35 is B=30 (A=3) or B=40 (A=4).
        # "ordinal" rank picks B=30 as rank-1 since distances are equal but
        # ordinal breaks ties by row appearance order.
        imp = KNNImputer(n_neighbors=1, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        # Either 3.0 or 4.0 is acceptable (tie at dist=5); just check it's finite
        assert result["A"][4] in {3.0, 4.0}

    def test_multiple_nulls_all_filled(self, X_multi_null):
        imp = KNNImputer(n_neighbors=1, subset=["A", "B"])
        imp.fit(X_multi_null)
        result = imp.transform(X_multi_null)
        assert result["A"].null_count() == 0
        assert result["B"].null_count() == 0

    def test_output_has_no_nulls_in_subset(self):
        X = pl.DataFrame(
            {
                "A": [1.0, None, 3.0, None, 5.0],
                "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X)
        result = imp.transform(X)
        assert result["A"].null_count() == 0

    def test_output_shape_unchanged(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        assert result.shape == X_simple.shape

    def test_non_subset_columns_unchanged(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        assert_frame_equal(result.select("B"), X_simple.select("B"))

    def test_fewer_training_rows_than_n_neighbors(self):
        # 2 complete training rows, n_neighbors=5 → uses all 2
        X = pl.DataFrame({"A": [1.0, 2.0, None], "B": [10.0, 20.0, 15.0]})
        imp = KNNImputer(n_neighbors=5, subset=["A"])
        imp.fit(X)
        result = imp.transform(X)
        # mean(1.0, 2.0) = 1.5
        assert result["A"][2] == pytest.approx(1.5)

    def test_fallback_to_global_median_when_no_features(self):
        # subset=["A", "B"], no other numeric cols → no feature cols for distance
        X = pl.DataFrame({"A": [1.0, 2.0, None], "B": [4.0, 6.0, None]})
        imp = KNNImputer(n_neighbors=2, subset=["A", "B"])
        imp.fit(X)
        result = imp.transform(X)
        # Global median of A (complete rows only: [1,2]) = 1.5
        # Global median of B (complete rows only: [4,6]) = 5.0
        assert result["A"][2] == pytest.approx(1.5)
        assert result["B"][2] == pytest.approx(5.0)

    def test_result_dtype_is_float(self, X_simple):
        imp = KNNImputer(n_neighbors=2, subset=["A"])
        imp.fit(X_simple)
        result = imp.transform(X_simple)
        assert result["A"].dtype == pl.Float64


# ---------------------------------------------------------------------------
# transform – distance weights
# ---------------------------------------------------------------------------


class TestTransformDistance:
    def test_distance_weights_closer_neighbor_dominates(self):
        # A=[1, 10, null], B=[1.0, 10.0, 2.0]
        # Distances from B=2: to B=1 → d=1, to B=10 → d=8
        # w0 = 1/(1+eps) ≈ 1.0,  w1 = 1/(8+eps) ≈ 0.125
        # weighted A = (1*1.0 + 10*0.125) / (1.0+0.125) = 2.25/1.125 = 2.0
        X = pl.DataFrame({"A": [1.0, 10.0, None], "B": [1.0, 10.0, 2.0]})
        imp = KNNImputer(n_neighbors=2, subset=["A"], weights="distance")
        imp.fit(X)
        result = imp.transform(X)
        assert result["A"][2] == pytest.approx(2.0, abs=1e-4)

    def test_distance_weights_differ_from_uniform(self):
        X = pl.DataFrame({"A": [1.0, 10.0, None], "B": [1.0, 10.0, 2.0]})

        imp_uniform = KNNImputer(n_neighbors=2, subset=["A"], weights="uniform")
        imp_uniform.fit(X)
        uniform_val = imp_uniform.transform(X)["A"][2]  # 5.5

        imp_dist = KNNImputer(n_neighbors=2, subset=["A"], weights="distance")
        imp_dist.fit(X)
        dist_val = imp_dist.transform(X)["A"][2]  # ≈ 2.0

        assert uniform_val != pytest.approx(dist_val, abs=0.01)

    def test_distance_weights_equal_distances_same_as_uniform(self):
        # When all neighbors are equidistant, weighted == uniform
        X = pl.DataFrame({"A": [1.0, 3.0, None], "B": [0.0, 0.0, 0.0]})
        imp = KNNImputer(n_neighbors=2, subset=["A"], weights="distance")
        imp.fit(X)
        result = imp.transform(X)
        # Both B=0, distances both 0+eps → equal weights → mean(1,3)=2
        assert result["A"][2] == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# sklearn API compatibility
# ---------------------------------------------------------------------------


class TestSklearnAPI:
    def test_get_params(self):
        imp = KNNImputer(n_neighbors=3, subset=["A"], weights="distance")
        params = imp.get_params()
        assert params["n_neighbors"] == 3
        assert params["subset"] == ["A"]
        assert params["weights"] == "distance"

    def test_set_params(self):
        imp = KNNImputer(n_neighbors=3)
        imp.set_params(n_neighbors=7)
        assert imp.n_neighbors == 7

    def test_set_params_returns_self(self):
        imp = KNNImputer()
        assert imp.set_params(n_neighbors=2) is imp

    def test_fit_returns_self(self):
        X = pl.DataFrame({"A": [1.0, None]})
        imp = KNNImputer(n_neighbors=1)
        assert imp.fit(X) is imp

    def test_fit_transform_equivalent_to_fit_then_transform(self, X_simple):
        imp1 = KNNImputer(n_neighbors=2, subset=["A"])
        result1 = imp1.fit_transform(X_simple)

        imp2 = KNNImputer(n_neighbors=2, subset=["A"])
        imp2.fit(X_simple)
        result2 = imp2.transform(X_simple)

        assert_frame_equal(result1, result2)

    def test_transform_empty_subset_returns_x_unchanged(self):
        """transform() before fit() (subset is None) returns X unchanged."""
        X = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        imp = KNNImputer(n_neighbors=2)
        # Do NOT call fit() — subset remains None → not self.subset is True
        result = imp.transform(X)
        assert_frame_equal(result, X)
