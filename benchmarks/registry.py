"""Registry of gators / scikit-learn / feature-engine transformer triples.

Only cases with a genuinely equivalent implementation in the other library(ies) are
listed here, so the comparison stays honest. When no equivalent exists (e.g. WOE
encoding in scikit-learn), the corresponding factory is ``None`` and the runner
reports "n/a" instead of a synthetic number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    kind: str  # "numeric" or "categorical"
    needs_y: bool
    gators_factory: Callable[[], object]
    sklearn_factory: Optional[Callable[[], object]]
    feature_engine_factory: Optional[Callable[[], object]]


def build_cases(numeric_cols: list[str], categorical_cols: list[str]) -> list[BenchmarkCase]:
    from gators.clippers import QuantileClipper
    from gators.discretizers import EqualSizeDiscretizer
    from gators.encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder, WOEEncoder
    from gators.imputers import NumericImputer
    from gators.scalers import StandardScaler

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        KBinsDiscretizer,
        OneHotEncoder as SkOneHotEncoder,
        OrdinalEncoder as SkOrdinalEncoder,
        StandardScaler as SkStandardScaler,
        TargetEncoder as SkTargetEncoder,
    )

    from feature_engine.discretisation import EqualFrequencyDiscretiser
    from feature_engine.encoding import (
        MeanEncoder as FeMeanEncoder,
        OneHotEncoder as FeOneHotEncoder,
        OrdinalEncoder as FeOrdinalEncoder,
        WoEEncoder as FeWoEEncoder,
    )
    from feature_engine.imputation import MeanMedianImputer
    from feature_engine.outliers import Winsorizer

    return [
        BenchmarkCase(
            name="NumericImputer (mean)",
            kind="numeric",
            needs_y=False,
            gators_factory=lambda: NumericImputer(strategy="mean", subset=numeric_cols, inplace=False),
            sklearn_factory=lambda: SimpleImputer(strategy="mean"),
            feature_engine_factory=lambda: MeanMedianImputer(
                imputation_method="mean", variables=numeric_cols
            ),
        ),
        BenchmarkCase(
            name="StandardScaler",
            kind="numeric",
            needs_y=False,
            gators_factory=lambda: StandardScaler(subset=numeric_cols, inplace=False),
            sklearn_factory=lambda: SkStandardScaler(),
            feature_engine_factory=None,  # feature-engine wraps sklearn's own implementation
        ),
        BenchmarkCase(
            name="QuantileClipper",
            kind="numeric",
            needs_y=False,
            gators_factory=lambda: QuantileClipper(subset=numeric_cols, inplace=False),
            sklearn_factory=None,  # no direct equivalent in scikit-learn
            feature_engine_factory=lambda: Winsorizer(
                capping_method="quantiles", tail="both", fold=0.01, variables=numeric_cols
            ),
        ),
        BenchmarkCase(
            name="EqualSizeDiscretizer (5 bins)",
            kind="numeric",
            needs_y=False,
            gators_factory=lambda: EqualSizeDiscretizer(num_bins=5, subset=numeric_cols, inplace=False),
            sklearn_factory=lambda: KBinsDiscretizer(
                n_bins=5, encode="ordinal", strategy="quantile", subsample=None
            ),
            feature_engine_factory=lambda: EqualFrequencyDiscretiser(q=5, variables=numeric_cols),
        ),
        BenchmarkCase(
            name="OneHotEncoder",
            kind="categorical",
            needs_y=False,
            gators_factory=lambda: OneHotEncoder(subset=categorical_cols),
            sklearn_factory=lambda: SkOneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            feature_engine_factory=lambda: FeOneHotEncoder(variables=categorical_cols),
        ),
        BenchmarkCase(
            name="OrdinalEncoder",
            kind="categorical",
            needs_y=False,
            gators_factory=lambda: OrdinalEncoder(subset=categorical_cols, inplace=False),
            sklearn_factory=lambda: SkOrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ),
            feature_engine_factory=lambda: FeOrdinalEncoder(
                encoding_method="arbitrary", variables=categorical_cols
            ),
        ),
        BenchmarkCase(
            name="TargetEncoder",
            kind="categorical",
            needs_y=True,
            gators_factory=lambda: TargetEncoder(subset=categorical_cols, inplace=False),
            sklearn_factory=lambda: SkTargetEncoder(),
            feature_engine_factory=lambda: FeMeanEncoder(variables=categorical_cols),
        ),
        BenchmarkCase(
            name="WOEEncoder",
            kind="categorical",
            needs_y=True,
            gators_factory=lambda: WOEEncoder(subset=categorical_cols, inplace=False),
            sklearn_factory=None,  # no WOE encoder in scikit-learn
            feature_engine_factory=lambda: FeWoEEncoder(variables=categorical_cols),
        ),
    ]
