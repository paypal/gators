#!/usr/bin/env python3.14
"""
STREAMLINED BENCHMARK APPROACH - Reduce Code by 80%
===================================================

Instead of repeating the same benchmark loop 6 times (one per encoder),
use a configuration dictionary and single loop.

BEFORE: ~300 lines (6 benchmarks × 50 lines each)
AFTER: ~50 lines (1 loop with configuration)

This is a TEMPLATE/REFERENCE file showing recommended patterns.
Not meant to be executed directly - copy patterns into actual benchmark notebooks.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import polars as pl
import pandas as pd
import numpy as np
import time
from typing import Dict, Optional

# Gators encoders
from gators.encoders import (
    OneHotEncoder,
    OrdinalEncoder,
    CountEncoder,
    RareCategoryEncoder,
    TargetEncoder,
    WOEEncoder
)

# Gators imputers
from gators.imputers import NumericImputer, StringImputer

# Gators discretizers
from gators.discretizers import (
    QuantileDiscretizer,
    EqualLengthDiscretizer,
    CustomDiscretizer,
    TreeBasedDiscretizer,
    GeometricDiscretizer
)

# Gators scalers
from gators.scalers import (
    StandardScaler,
    MinmaxScaler,
    BoxCox,
    YeoJohnson,
    LogScaler,
    PowerScaler,
    ArcSinSquareRootScaler
)

# Feature-engine encoders
from feature_engine.encoding import (
    OneHotEncoder as FEOneHotEncoder,
    OrdinalEncoder as FEOrdinalEncoder,
    CountFrequencyEncoder,
    RareLabelEncoder,
    MeanEncoder,
    WoEEncoder
)

# Feature-engine imputers
from feature_engine.imputation import (
    MeanMedianImputer,
    ArbitraryNumberImputer,
    CategoricalImputer
)

# Feature-engine discretisers
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    GeometricWidthDiscretiser
)

# Feature-engine transformers (for scalers)
from feature_engine.transformation import (
    BoxCoxTransformer,
    YeoJohnsonTransformer,
    LogTransformer,
    PowerTransformer,
    ArcsinTransformer
)

# Sklearn scalers (feature-engine doesn't have standard/minmax scalers)
import matplotlib.pyplot as plt
import seaborn as sns

# Import benchmark utilities from benchmark.py (includes timeout support)
from benchmarks.benchmark import benchmark_transformer, DEFAULT_TIMEOUT


def benchmark_all_transformers(transformer_configs, datasets, dataset_sizes):
    """Generic function to benchmark any set of transformers across dataset sizes.
    
    Parameters
    ----------
    transformer_configs : dict
        Configuration for each transformer type.
    datasets : dict
        Generated datasets by size.
    dataset_sizes : list
        List of dataset sizes to test.
    n_columns : int, default=25
        Fixed number of columns to use for benchmarking.

    Returns
    -------
    pd.DataFrame
        All benchmark results with transformer_type column.
    """
    all_results = []
    
    for transformer_name, config in transformer_configs.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking {transformer_name}")
        print(f"{'='*70}")
        
        for size in dataset_sizes:
            # Get dataset
            X_polars = datasets[size]['polars']
            X_pandas = datasets[size]['pandas']
            
            # Create transformers
            gators_transformer = config['gators_class'](**config['gators_params'])
            fe_transformer = config['fe_class'](**config['fe_params'])
            
            # Prepare benchmark kwargs
            benchmark_kwargs = {'n_runs': 3, 'timeout_seconds': DEFAULT_TIMEOUT}
            if config.get('supervised', False):
                benchmark_kwargs['y_polars'] = datasets[size]['y_polars']
                benchmark_kwargs['y_pandas'] = datasets[size]['y_pandas']
            
            # Benchmark
            results = benchmark_transformer(gators_transformer, fe_transformer, 
                                           X_polars, X_pandas, **benchmark_kwargs)
            
            # Store
            all_results.append({
                config.get('type_column', 'transformer_type'): transformer_name,
                'dataset_size': size,
                **{k: v for k, v in results.items()}  # Unpack all results
            })
            
            # Print progress with timeout indicator
            speedup_str = f"≥{results['speedup_total']:.2f}x" if results.get('timed_out', False) else f"{results['speedup_total']:.2f}x"
            print(f"    {size:>8,} rows: {speedup_str} speedup")
    
    return pd.DataFrame(all_results)





# # ============================================================================
# # APPROACH 3: Imputers Configuration
# # ============================================================================

# imputers_config = {
#     'Mean': {
#         'gators_class': NumericImputer,
#         'gators_params': {'strategy': 'mean', 'inplace': True},
#         'fe_class': MeanMedianImputer,
#         'fe_params': {'imputation_method': 'mean'},
#         'column_type': 'numeric',  # Use numeric columns
#         'supervised': False
#     },
#     'Median': {
#         'gators_class': NumericImputer,
#         'gators_params': {'strategy': 'median', 'inplace': True},
#         'fe_class': MeanMedianImputer,
#         'fe_params': {'imputation_method': 'median'},
#         'column_type': 'numeric',  # Use numeric columns
#         'supervised': False
#     },
#     'Constant_Numeric': {
#         'gators_class': NumericImputer,
#         'gators_params': {'strategy': 'constant', 'value': 0.0, 'inplace': True},
#         'fe_class': ArbitraryNumberImputer,
#         'fe_params': {'arbitrary_number': 0.0},
#         'column_type': 'numeric',  # Use numeric columns
#         'supervised': False
#     },
#     'Constant_Categorical': {
#         'gators_class': StringImputer,
#         'gators_params': {'strategy': 'constant', 'value': 'MISSING', 'inplace': True},
#         'fe_class': CategoricalImputer,
#         'fe_params': {'imputation_method': 'missing', 'fill_value': 'MISSING'},
#         'column_type': 'categorical',  # Use categorical/string columns
#         'supervised': False
#     }
# }

# # SINGLE LOOP to benchmark all imputers
# # Note: Imputers don't have cardinality groups, just column types (numeric vs categorical)
# all_imputer_results = []

# for imputer_name, config in imputers_config.items():
#     print(f"\n{'='*70}")
#     print(f"Benchmarking {imputer_name} Imputer")
#     print(f"{'='*70}")
    
#     for size in dataset_sizes:
#         X_polars = datasets[size]['polars']
#         X_pandas = datasets[size]['pandas']
        
#         # Select columns based on type
#         if config['column_type'] == 'numeric':
#             subset_cols = [col for col in X_polars.columns if col.startswith('num_')]
#         else:  # categorical
#             subset_cols = [col for col in X_polars.columns if col.startswith('str_')]
        
#         # Create transformers
#         gators_imp = config['gators_class'](subset=subset_cols, **config['gators_params'])
#         fe_imp = config['fe_class'](variables=subset_cols, **config['fe_params'])
        
#         # Benchmark
#         results = benchmark_transformer(
#             gators_imp,
#             fe_imp,
#             X_polars.select(subset_cols),
#             X_pandas[subset_cols],
#             n_runs=3,
#             timeout_seconds=DEFAULT_TIMEOUT
#         )
        
#         # Store results
#         all_imputer_results.append({
#             'imputer_type': imputer_name,
#             'dataset_size': size,
#             'n_columns': len(subset_cols),
#             'column_type': config['column_type'],
#             'gators_fit': results['gators_fit'],
#             'gators_transform': results['gators_transform'],
#             'gators_total': results['gators_total'],
#             'fe_fit': results['comparison_fit'],
#             'fe_transform': results['comparison_transform'],
#             'fe_total': results['comparison_total'],
#             'speedup_total': results['speedup_total']
#         })
        
#         print(f"  {size:>8,} rows ({len(subset_cols)} cols): Gators={results['gators_total']:.4f}s, "
#               f"FE={results['comparison_total']:.4f}s, "
#               f"Speedup={results['speedup_total']:.2f}x")

# # Convert to DataFrame (already has imputer_type column!)
# all_imputer_results = pd.DataFrame(all_imputer_results)
# print(f"\n✅ All {len(imputers_config)} imputers benchmarked: {len(all_imputer_results)} total runs")


# # ============================================================================
# # DISCRETIZERS CONFIGURATION
# # ============================================================================

# discretizers_config = {
#     'EqualFrequency': {
#         'gators_class': QuantileDiscretizer,
#         'gators_params': {'num_bins': 5, 'inplace': True, 'drop_columns': True},
#         'fe_class': EqualFrequencyDiscretiser,
#         'fe_params': {'q': 5, 'return_object': False, 'return_boundaries': False},
#         'supervised': False
#     },
#     'EqualWidth': {
#         'gators_class': EqualLengthDiscretizer,
#         'gators_params': {'num_bins': 5, 'inplace': True, 'drop_columns': True},
#         'fe_class': EqualWidthDiscretiser,
#         'fe_params': {'bins': 5, 'return_object': False, 'return_boundaries': False},
#         'supervised': False
#     },
#     'Arbitrary': {
#         'gators_class': CustomDiscretizer,
#         'gators_params': {
#             'bins': {
#                 'num_0': [-np.inf, 0.25, 0.5, 0.75, np.inf],
#                 'num_1': [-np.inf, 0.25, 0.5, 0.75, np.inf],
#                 'num_2': [-np.inf, 0.25, 0.5, 0.75, np.inf]
#             },
#             'num_bins': 4,
#             'inplace': True,
#             'drop_columns': True
#         },
#         'fe_class': ArbitraryDiscretiser,
#         'fe_params': {
#             'binning_dict': {
#                 'num_0': [-np.inf, 0.25, 0.5, 0.75, np.inf],
#                 'num_1': [-np.inf, 0.25, 0.5, 0.75, np.inf],
#                 'num_2': [-np.inf, 0.25, 0.5, 0.75, np.inf]
#             },
#             'return_object': False,
#             'return_boundaries': False
#         },
#         'supervised': False,
#         'needs_custom_bins': True  # Flag that this needs bin configuration per dataset
#     },
#     'DecisionTree': {
#         'gators_class': TreeBasedDiscretizer,
#         'gators_params': {
#             'num_bins': 5,
#             'task': 'classification',
#             'min_samples_leaf': 10,
#             'inplace': True,
#             'drop_columns': True
#         },
#         'fe_class': DecisionTreeDiscretiser,
#         'fe_params': {
#             'cv': 3,
#             'scoring': 'roc_auc',
#             'regression': False,
#             'return_object': False,
#             'return_boundaries': False
#         },
#         'supervised': True  # Needs y
#     },
#     'Geometric': {
#         'gators_class': GeometricDiscretizer,
#         'gators_params': {'num_bins': 5, 'inplace': True, 'drop_columns': True},
#         'fe_class': GeometricWidthDiscretiser,
#         'fe_params': {'bins': 5, 'return_object': False, 'return_boundaries': False},
#         'supervised': False
#     }
# }

# # SINGLE LOOP to benchmark all discretizers
# all_discretizer_results = []

# # Get numeric columns (discretizers work on numeric data)
# numeric_cols = [col for col in datasets[1_000]['polars'].columns if col.startswith('num_')][:3]  # Use first 3 for testing

# for discretizer_name, config in discretizers_config.items():
#     print(f"\n{'='*70}")
#     print(f"Benchmarking {discretizer_name} Discretizer")
#     print(f"{'='*70}")
    
#     for size in dataset_sizes:
#         X_polars = datasets[size]['polars']
#         X_pandas = datasets[size]['pandas']
        
#         # Handle arbitrary discretizer that needs custom bins per column
#         gators_params = config['gators_params'].copy()
#         fe_params = config['fe_params'].copy()
        
#         # Create transformers
#         gators_disc = config['gators_class'](subset=numeric_cols, **gators_params)
#         fe_disc = config['fe_class'](variables=numeric_cols, **fe_params)
        
#         # Prepare benchmark kwargs
#         benchmark_kwargs = {'n_runs': 3, 'timeout_seconds': DEFAULT_TIMEOUT}
        
#         # Add y for supervised discretizers
#         if config.get('supervised', False):
#             benchmark_kwargs['y_polars'] = datasets[size]['y_polars']
#             benchmark_kwargs['y_pandas'] = datasets[size]['y_pandas']
        
#         # Benchmark
#         results = benchmark_transformer(
#             gators_disc,
#             fe_disc,
#             X_polars.select(numeric_cols),
#             X_pandas[numeric_cols],
#             **benchmark_kwargs
#         )
        
#         # Store results
#         all_discretizer_results.append({
#             'discretizer_type': discretizer_name,
#             'dataset_size': size,
#             'n_columns': len(numeric_cols),
#             'gators_fit': results['gators_fit'],
#             'gators_transform': results['gators_transform'],
#             'gators_total': results['gators_total'],
#             'fe_fit': results['comparison_fit'],
#             'fe_transform': results['comparison_transform'],
#             'fe_total': results['comparison_total'],
#             'speedup_total': results['speedup_total']
#         })
        
#         print(f"  {size:>8,} rows ({len(numeric_cols)} cols): Gators={results['gators_total']:.4f}s, "
#               f"FE={results['comparison_total']:.4f}s, "
#               f"Speedup={results['speedup_total']:.2f}x")

# # Convert to DataFrame (already has discretizer_type column!)
# all_discretizer_results = pd.DataFrame(all_discretizer_results)
# print(f"\n✅ All {len(discretizers_config)} discretizers benchmarked: {len(all_discretizer_results)} total runs")


# # ============================================================================
# # SCALERS CONFIGURATION
# # ============================================================================

# scalers_config = {
#     'StandardScaler': {
#         'gators_class': StandardScaler,
#         'gators_params': {'drop_columns': True},
#         'fe_class': SklearnStandardScaler,  # sklearn, not feature-engine
#         'fe_params': {},
#         'supervised': False,
#         'comparison_lib': 'sklearn'
#     },
#     'MinmaxScaler': {
#         'gators_class': MinmaxScaler,
#         'gators_params': {'drop_columns': True},
#         'fe_class': SklearnMinMaxScaler,  # sklearn, not feature-engine
#         'fe_params': {},
#         'supervised': False,
#         'comparison_lib': 'sklearn'
#     },
#     'BoxCox': {
#         'gators_class': BoxCox,
#         'gators_params': {'drop_columns': False},  # Needs lambdas parameter
#         'fe_class': BoxCoxTransformer,
#         'fe_params': {},
#         'supervised': False,
#         'needs_lambdas': True,  # Requires pre-computed lambda values
#         'positive_only': True,  # Only works with positive values
#         'comparison_lib': 'feature-engine'
#     },
#     'YeoJohnson': {
#         'gators_class': YeoJohnson,
#         'gators_params': {'drop_columns': False},  # Needs lambdas parameter
#         'fe_class': YeoJohnsonTransformer,
#         'fe_params': {},
#         'supervised': False,
#         'needs_lambdas': True,  # Requires pre-computed lambda values
#         'comparison_lib': 'feature-engine'
#     },
#     'LogScaler': {
#         'gators_class': LogScaler,
#         'gators_params': {'drop_columns': True},
#         'fe_class': LogTransformer,
#         'fe_params': {},
#         'supervised': False,
#         'positive_only': True,  # Only works with positive values
#         'comparison_lib': 'feature-engine'
#     },
#     'PowerScaler': {
#         'gators_class': PowerScaler,
#         'gators_params': {'power': 2, 'drop_columns': True},
#         'fe_class': PowerTransformer,
#         'fe_params': {'exp': 2},
#         'supervised': False,
#         'comparison_lib': 'feature-engine'
#     },
#     'ArcSinSquareRoot': {
#         'gators_class': ArcSinSquareRootScaler,
#         'gators_params': {'drop_columns': True},
#         'fe_class': ArcsinTransformer,
#         'fe_params': {},
#         'supervised': False,
#         'comparison_lib': 'feature-engine'
#     }
# }

# # SINGLE LOOP to benchmark all scalers
# all_scaler_results = []

# # Get numeric columns (scalers work on numeric data)
# numeric_cols_scalers = [col for col in datasets[1_000]['polars'].columns if col.startswith('num_')][:5]  # Use first 5 for testing

# for scaler_name, config in scalers_config.items():
#     print(f"\n{'='*70}")
#     print(f"Benchmarking {scaler_name}")
#     print(f"{'='*70}")
    
#     for size in dataset_sizes:
#         X_polars = datasets[size]['polars']
#         X_pandas = datasets[size]['pandas']
        
#         # Handle special cases for transformations that need lambdas or positive values
#         if config.get('positive_only', False):
#             # Select only positive columns (exponential distributions)
#             pos_cols = [col for col in numeric_cols_scalers if 'exp_' in col or X_polars[col].min() > 0]
#             if not pos_cols:
#                 print(f"  Skipping {scaler_name} - no positive columns available")
#                 continue
#             X_polars_subset = X_polars.select(pos_cols)
#             X_pandas_subset = X_pandas[pos_cols]
#             subset = pos_cols
#         else:
#             X_polars_subset = X_polars.select(numeric_cols_scalers)
#             X_pandas_subset = X_pandas[numeric_cols_scalers]
#             subset = numeric_cols_scalers
        
#         # Handle transformers that need lambda values (BoxCox, YeoJohnson)
#         gators_params = config['gators_params'].copy()
#         if config.get('needs_lambdas', False):
#             # Compute lambdas using scipy (would be done in actual benchmark)
#             from scipy import stats
#             lambdas = {}
#             for col in subset:
#                 if scaler_name == 'BoxCox':
#                     _, fitted_lambda = stats.boxcox(X_pandas_subset[col].values)
#                 else:  # YeoJohnson
#                     _, fitted_lambda = stats.yeojohnson(X_pandas_subset[col].values)
#                 lambdas[col] = fitted_lambda
#             gators_params['lambdas'] = lambdas
        
#         # Create transformers
#         gators_scaler = config['gators_class'](subset=subset, **gators_params)
        
#         # Feature-engine uses 'variables' parameter, sklearn doesn't
#         if config['comparison_lib'] == 'feature-engine':
#             fe_scaler = config['fe_class'](variables=subset, **config['fe_params'])
#         else:  # sklearn
#             fe_scaler = config['fe_class'](**config['fe_params'])
        
#         # Benchmark
#         results = benchmark_transformer(
#             gators_scaler,
#             fe_scaler,
#             X_polars_subset,
#             X_pandas_subset,
#             n_runs=3,
#             timeout_seconds=DEFAULT_TIMEOUT
#         )
        
#         # Store results
#         all_scaler_results.append({
#             'scaler_type': scaler_name,
#             'dataset_size': size,
#             'n_columns': len(subset),
#             'comparison_lib': config['comparison_lib'],
#             'gators_fit': results['gators_fit'],
#             'gators_transform': results['gators_transform'],
#             'gators_total': results['gators_total'],
#             'comparison_fit': results['comparison_fit'],
#             'comparison_transform': results['comparison_transform'],
#             'comparison_total': results['comparison_total'],
#             'speedup_total': results['speedup_total']
#         })
        
#         comp_lib = config['comparison_lib']
#         print(f"  {size:>8,} rows ({len(subset)} cols): Gators={results['gators_total']:.4f}s, "
#               f"{comp_lib}={results['comparison_total']:.4f}s, "
#               f"Speedup={results['speedup_total']:.2f}x")

# # Convert to DataFrame (already has scaler_type column!)
# all_scaler_results = pd.DataFrame(all_scaler_results)
# print(f"\n✅ All {len(scalers_config)} scalers benchmarked: {len(all_scaler_results)} total runs")


# # ============================================================================
# # APPROACH 4: For Imputers/Scalers (No Cardinality Groups) - LEGACY
# # ============================================================================

# imputer_configs = {
#     'Mean': {
#         'gators_class': NumericImputer,
#         'gators_params': {'strategy': 'mean', 'inplace': True},
#         'fe_class': MeanMedianImputer,
#         'fe_params': {'imputation_method': 'mean'},
#         'column_selector': lambda df: [c for c in df.columns if c.startswith('num_')],
#         'type_column': 'imputer_type'
#     },
#     'Median': {
#         'gators_class': NumericImputer,
#         'gators_params': {'strategy': 'median', 'inplace': True},
#         'fe_class': MeanMedianImputer,
#         'fe_params': {'imputation_method': 'median'},
#         'column_selector': lambda df: [c for c in df.columns if c.startswith('num_')],
#         'type_column': 'imputer_type'
#     },
#     # ... etc
# }

# # Simplified loop for imputers (no cardinality dimension)
# all_results = []
# for imputer_name, config in imputer_configs.items():
#     print(f"\nBenchmarking {imputer_name} Imputer")
    
#     # Get columns once (same for all sizes)
#     cols = config['column_selector'](datasets[1_000]['polars'])
    
#     for size in dataset_sizes:
#         gators_imp = config['gators_class'](subset=cols, **config['gators_params'])
#         fe_imp = config['fe_class'](variables=cols, **config['fe_params'])
        
#         results = benchmark_transformer(
#             gators_imp, fe_imp,
#             datasets[size]['polars'].select(cols),
#             datasets[size]['pandas'][cols],
#             n_runs=3, timeout_seconds=DEFAULT_TIMEOUT
#         )
        
#         all_results.append({
#             'imputer_type': imputer_name,
#             'dataset_size': size,
#             'n_columns': len(cols),
#             **results
#         })

# all_results = pd.DataFrame(all_results)


# # ============================================================================
# # KEY BENEFITS
# # ============================================================================
# """
# 1. **80% Code Reduction**: 300 lines → 50 lines
# 2. **DRY Principle**: Define once, use everywhere
# 3. **Easy to Add**: New encoder? Just add to config dict
# 4. **Maintainable**: Change logic once, applies to all
# 5. **Type Safety**: Clear structure with configs
# 6. **Immediate DataFrame**: all_results already has type column
# 7. **Flexible**: Easy to add new parameters or filters
# 8. **Reusable**: Same approach for encoders, imputers, scalers

# CONS:
# - Slightly less explicit than separate sections
# - Harder to debug individual transformers
# - Lost section-by-section narrative in notebooks

# RECOMMENDATION:
# Use configuration approach for production/analysis.
# Keep separate sections for tutorial/documentation notebooks.
# """
