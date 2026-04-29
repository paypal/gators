"""Benchmark utilities for Gators performance testing.

This module provides deterministic data generation and benchmarking functions
for comparing Gators transformers against feature-engine and sklearn.

All data generation is deterministic (no randomness) for reproducible results.
"""

import polars as pl
import pandas as pd
import numpy as np
import time
from typing import Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
import signal
from contextlib import contextmanager
import matplotlib.pyplot as plt


# ============================================================================
# CONSTANTS
# ============================================================================

DATASET_SIZES = [1_000, 10_000, 100_000, 1_000_000]

DEFAULT_TIMEOUT = 300  # 3 minutes for testing

DEFAULT_N_RUNS = 3

DEFAULT_WARMUP = True


# ============================================================================
# TIMEOUT UTILITIES
# ============================================================================

class TimeoutError(Exception):
    """Raised when a benchmark operation exceeds the timeout."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for setting a timeout on operations.
    
    Parameters
    ----------
    seconds : int
        Maximum time allowed for the operation.
        
    Raises
    ------
    TimeoutError
        If the operation exceeds the timeout.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# DATA GENERATION FUNCTIONS (DETERMINISTIC)
# ============================================================================

def generate_datetime_data(
    n_rows: int,
    n_datetime_cols: int = 3
) -> Tuple[pl.DataFrame, pd.DataFrame]:
    """Generate deterministic dataset with datetime columns.
    
    Creates evenly-spaced timestamps spanning 2020-2023.
    No randomness - results are fully reproducible.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    n_datetime_cols : int, default=3
        Number of datetime columns to create.
        
    Returns
    -------
    Tuple[pl.DataFrame, pd.DataFrame]
        Polars and pandas DataFrames with identical datetime data.
    """
    data_pandas = {}
    data_polars = {}
    
    # Generate datetime columns spanning 2020-2023
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range_seconds = int((end_date - start_date).total_seconds())
    
    for i in range(n_datetime_cols):
        # Generate evenly-spaced timestamps (deterministic)
        # Offset each column slightly to create variation
        offset_seconds = i * 3600  # 1 hour offset per column
        step_seconds = date_range_seconds // n_rows
        
        timestamps = [
            start_date + timedelta(seconds=offset_seconds + j * step_seconds)
            for j in range(n_rows)
        ]
        
        data_pandas[f'datetime_{i}'] = pd.to_datetime(timestamps)
        data_polars[f'datetime_{i}'] = timestamps
    
    # Create DataFrames
    df_pandas = pd.DataFrame(data_pandas)
    df_polars = pl.DataFrame(data_polars)
    
    return df_polars, df_pandas


def generate_numeric_data(
    n_rows: int,
    n_cols: int = 10
) -> Tuple[pl.DataFrame, pd.DataFrame]:
    """Generate deterministic numeric data with various patterns.
    
    Creates numeric columns with different deterministic patterns:
    - Linear sequences
    - Quadratic sequences
    - Exponential-like sequences (always positive for BoxCox)
    - Periodic sequences
    
    No randomness - results are fully reproducible.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    n_cols : int, default=10
        Number of numeric columns to create.
        
    Returns
    -------
    Tuple[pl.DataFrame, pd.DataFrame]
        Polars and pandas DataFrames with identical numeric data.
    """
    data = {}
    
    # Generate indices for deterministic patterns
    indices = np.arange(n_rows)
    
    for i in range(n_cols):
        if i % 4 == 0:
            # Linear pattern: 0 to 1000
            data[f'linear_{i}'] = np.linspace(0, 1000, n_rows)
        elif i % 4 == 1:
            # Quadratic pattern
            data[f'quadratic_{i}'] = (indices / n_rows) ** 2 * 1000
        elif i % 4 == 2:
            # Exponential-like pattern (always positive for BoxCox)
            data[f'exp_{i}'] = np.exp(indices / n_rows * 5) + 1
        else:
            # Periodic pattern
            data[f'periodic_{i}'] = 500 + 400 * np.sin(indices / n_rows * 2 * np.pi)
    
    # Create DataFrames
    df_pandas = pd.DataFrame(data)
    df_polars = pl.DataFrame(data)
    
    return df_polars, df_pandas


def generate_categorical_data(
    n_rows: int,
    num_categories: int = 5,
    n_low_cardinality: int = 3,
    n_medium_cardinality: int = 3,
    n_high_cardinality: int = 2
) -> Tuple[pl.DataFrame, pd.DataFrame]:
    """Generate deterministic categorical data with varying cardinality levels.
    
    Creates categorical columns with repeating patterns.
    No randomness - results are fully reproducible.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    num_categories : int, default=5
        Number of unique categories for simple mode.
        Used when n_low_cardinality=n_medium_cardinality=n_high_cardinality=0
        Common values: 5, 50, 500
    n_low_cardinality : int, default=3
        Number of low cardinality columns to create (5-10 categories).
    n_medium_cardinality : int, default=3
        Number of medium cardinality columns to create (20-50 categories).
    n_high_cardinality : int, default=2
        Number of high cardinality columns to create (100-250 categories).
        
    Returns
    -------
    Tuple[pl.DataFrame, pd.DataFrame]
        Polars and pandas DataFrames with identical categorical data.
    """
    data_pandas = {}
    data_polars = {}
    
    # If all cardinality counts are 0, use simple mode with num_categories
    if n_low_cardinality == 0 and n_medium_cardinality == 0 and n_high_cardinality == 0:
        categories = [f'cat_{i}' for i in range(num_categories)]
        for col_idx in range(3):
            offset = col_idx * (num_categories // 3) if num_categories >= 3 else col_idx
            values = [categories[(i + offset) % num_categories] for i in range(n_rows)]
            data_pandas[f'category_{col_idx}'] = values
            data_polars[f'category_{col_idx}'] = values
    else:
        # Low cardinality columns (5-10 categories) - deterministic
        cardinality_levels = [5, 7, 10]  # Fixed cardinalities for reproducibility
        for i in range(n_low_cardinality):
            n_categories = cardinality_levels[i % len(cardinality_levels)]
            categories = [f'low_{i}_cat_{j}' for j in range(n_categories)]
            # Cycle through categories deterministically
            values = [categories[(idx + i) % n_categories] for idx in range(n_rows)]
            data_pandas[f'low_card_{i}'] = values
            data_polars[f'low_card_{i}'] = values
        
        # Medium cardinality columns (20-50 categories) - deterministic
        cardinality_levels = [20, 35, 50]
        for i in range(n_medium_cardinality):
            n_categories = cardinality_levels[i % len(cardinality_levels)]
            categories = [f'med_{i}_cat_{j}' for j in range(n_categories)]
            # Cycle through categories deterministically with different pattern
            values = [categories[(idx * 3 + i) % n_categories] for idx in range(n_rows)]
            data_pandas[f'med_card_{i}'] = values
            data_polars[f'med_card_{i}'] = values
        
        # High cardinality columns (100-250 categories) - deterministic
        cardinality_levels = [100, 175, 250]
        for i in range(n_high_cardinality):
            n_categories = cardinality_levels[i % len(cardinality_levels)]
            categories = [f'high_{i}_cat_{j}' for j in range(n_categories)]
            # Cycle through categories deterministically with different pattern
            values = [categories[(idx * 7 + i * 13) % n_categories] for idx in range(n_rows)]
            data_pandas[f'high_card_{i}'] = values
            data_polars[f'high_card_{i}'] = values
    
    # Create DataFrames
    df_pandas = pd.DataFrame(data_pandas)
    df_polars = pl.DataFrame(data_polars)
    
    return df_polars, df_pandas


def generate_binary_target(
    n_rows: int,
    positive_rate: float = 0.3
) -> Tuple[pl.Series, pd.Series]:
    """Generate deterministic binary target variable.
    
    Creates a binary target (0/1) with specified positive rate.
    No randomness - results are fully reproducible.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    positive_rate : float, default=0.3
        Proportion of positive cases (value=1).
        
    Returns
    -------
    Tuple[pl.Series, pd.Series]
        Polars and pandas Series with identical binary target data.
    """
    # Create deterministic binary pattern
    # Every Nth row is positive, where N = 1/positive_rate
    if positive_rate > 0 and positive_rate < 1:
        period = int(1.0 / positive_rate)
        values = [(i % period < 1) for i in range(n_rows)]
    elif positive_rate >= 1:
        values = [1] * n_rows
    else:
        values = [0] * n_rows
    
    # Convert to int
    values = [int(v) for v in values]
    
    # Create Series
    y_pandas = pd.Series(values, name='target')
    y_polars = pl.Series('target', values)
    
    return y_polars, y_pandas




def generate_data_with_missing(
    n_rows: int,
    n_numeric: int = 50,
    n_categorical: int = 10,
    num_categories: int = 10
) -> Tuple[pl.DataFrame, pd.DataFrame]:
    """Generate deterministic data with missing values for imputation testing.
    
    Creates numeric and categorical columns with systematic missing patterns.
    No randomness - results are fully reproducible.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    n_numeric : int, default=10
        Number of numeric columns to create.
    n_categorical : int, default=5
        Number of categorical columns to create.
    num_categories : int, default=5
        Number of unique categories per categorical column.
        
    Returns
    -------
    Tuple[pl.DataFrame, pd.DataFrame]
        Polars and pandas DataFrames with identical data including missing values.
    """
    data_pandas = {}
    data_polars = {}
    
    # Generate indices for deterministic patterns
    indices = np.arange(n_rows)
    
    # Numeric columns with missing values (every 7th value is missing)
    for i in range(n_numeric):
        # Create deterministic pattern
        values = (indices / n_rows) ** (1 + i * 0.1) * 1000
        
        # Insert missing values at deterministic positions (every 7th + offset)
        mask = (indices + i) % 7 == 0
        values_with_missing = values.copy()
        values_with_missing[mask] = np.nan
        
        data_pandas[f'num_{i}'] = values_with_missing
        data_polars[f'num_{i}'] = values_with_missing
    
    # Categorical columns with missing values (every 10th value is missing)
    categories = [f'cat_{i}' for i in range(num_categories)]
    
    for col_idx in range(n_categorical):
        # Cycle through categories deterministically
        offset = col_idx * (num_categories // 3) if num_categories >= 3 else col_idx
        values = [categories[(i + offset) % num_categories] for i in range(n_rows)]
        
        # Insert missing values at deterministic positions (every 10th + offset)
        values_with_missing_pandas = []
        values_with_missing_polars = []
        
        for i, val in enumerate(values):
            if (i + col_idx) % 10 == 0:
                values_with_missing_pandas.append(None)
                values_with_missing_polars.append(None)
            else:
                values_with_missing_pandas.append(val)
                values_with_missing_polars.append(val)
        
        data_pandas[f'str_{col_idx}'] = values_with_missing_pandas
        data_polars[f'str_{col_idx}'] = values_with_missing_polars
    
    # Create DataFrames
    df_pandas = pd.DataFrame(data_pandas)
    df_polars = pl.DataFrame(data_polars)
    
    return df_polars, df_pandas


def generate_num_datasets(dataset_sizes=[1_000, 10_000, 100_000, 1_000_000]):
    datasets = {}
    for size in dataset_sizes:
        print(f"Generating {size:,} row dataset...")
        df_polars, df_pandas = generate_numeric_data(n_rows=size, n_cols=50)

        # Generate binary target for supervised encoders (deterministic)
        y_polars, y_pandas = generate_binary_target(n_rows=size, positive_rate=0.3)
        datasets[size] = {
            'polars': df_polars,
            'pandas': df_pandas,
            'y_polars': y_polars,
            'y_pandas': y_pandas
        }
    return datasets


def generate_cat_datasets(dataset_sizes=[1_000, 10_000, 100_000, 1_000_000]):
    datasets = {}
    for size in dataset_sizes:
        print(f"Generating {size:,} row dataset...")
        df_polars, df_pandas = generate_categorical_data(n_rows=size, n_low_cardinality=20, n_medium_cardinality=10, n_high_cardinality=10)

        # Generate binary target for supervised encoders (deterministic)
        y_polars, y_pandas = generate_binary_target(n_rows=size, positive_rate=0.3)
        datasets[size] = {
            'polars': df_polars,
            'pandas': df_pandas,
            'y_polars': y_polars,
            'y_pandas': y_pandas
        }
    return datasets


def generate_datasets(dataset_sizes=[1_000, 10_000, 100_000, 1_000_000]):
    datasets = {}
    for size in dataset_sizes:
        print(f"Generating {size:,} row dataset...")
        df_polars_num, df_pandas_num = generate_numeric_data(n_rows=size, n_cols=50)

        df_polars_cat, df_pandas_cat = generate_categorical_data(
            n_rows=size,
            n_low_cardinality=5,
            n_medium_cardinality=5,
            n_high_cardinality=5
        )
        # df_polars_dt, df_pandas_dt = generate_datetime_data(n_rows=size, n_datetime_cols=5)
        
        # Generate binary target for supervised encoders (deterministic)
        y_polars, y_pandas = generate_binary_target(n_rows=size, positive_rate=0.3)
        
        datasets[size] = {
            'polars': pl.concat([df_polars_num, df_polars_cat], how='horizontal'),
            'pandas': pd.concat([df_pandas_num, df_pandas_cat], axis=1),
            'y_polars': y_polars,
            'y_pandas': y_pandas
        }
    return datasets


def generate_datasets_with_missing(dataset_sizes=[1_000, 10_000, 100_000, 1_000_000]):
    datasets = {}
    for size in dataset_sizes:
        print(f"Generating {size:,} row dataset...")
        df_polars, df_pandas = generate_data_with_missing(
            n_rows=size,
            n_numeric=25,
            n_categorical=15,
            num_categories=25
        )
        datasets[size] = {'polars': df_polars, 'pandas': df_pandas}
    return datasets


# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================

def benchmark_transformer(
    gators_transformer: Any,
    comparison_transformer: Any,
    X_polars: pl.DataFrame,
    X_pandas: pd.DataFrame,
    y_polars: pl.Series = None,
    y_pandas: pd.Series = None,
    n_runs: int = DEFAULT_N_RUNS,
    warmup: bool = DEFAULT_WARMUP,
    timeout_seconds: int = DEFAULT_TIMEOUT
) -> Dict[str, float]:
    """Benchmark fit and transform times for both transformers.
    
    Compares a Gators transformer (Polars-based) against a comparison
    transformer (typically feature-engine or sklearn, pandas-based).
    
    Parameters
    ----------
    gators_transformer : Any
        Gators transformer instance (uses Polars DataFrames).
    comparison_transformer : Any
        Comparison transformer instance (uses pandas DataFrames).
        Typically from feature-engine or sklearn.
    X_polars : pl.DataFrame
        Input Polars DataFrame for Gators transformer.
    X_pandas : pd.DataFrame
        Input pandas DataFrame for comparison transformer.
    y_polars : pl.Series, optional
        Target variable for supervised transformers (Polars Series).
    y_pandas : pd.Series, optional
        Target variable for supervised transformers (pandas Series).
    n_runs : int, default=3
        Number of benchmark runs to perform (median is used).
    warmup : bool, default=True
        Whether to perform warmup runs to avoid JIT compilation effects.
    timeout_seconds : int, default=180
        Maximum time allowed for each fit/transform operation (3 minutes).
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing timing results:
        - gators_fit: Gators fit time (seconds)
        - gators_transform: Gators transform time (seconds)
        - gators_total: Total Gators time (seconds)
        - comparison_fit: Comparison library fit time (seconds, or timeout if timed out)
        - comparison_transform: Comparison library transform time (seconds, or timeout if timed out)
        - comparison_total: Total comparison time (seconds, or timeout if timed out)
        - speedup_fit: Fit speedup ratio (comparison/gators)
        - speedup_transform: Transform speedup ratio (comparison/gators)
        - speedup_total: Total speedup ratio (comparison/gators, at least timeout/gators if timed out)
        - timed_out: Boolean indicating if comparison timed out
        
    Raises
    ------
    TimeoutError
        If Gators operations exceed the timeout (comparison timeouts are handled gracefully).
    """
    results = {}
    
    # Warmup runs (avoid JIT compilation effects)
    if warmup:
        try:
            with timeout(timeout_seconds):
                if y_polars is not None:
                    gators_transformer.fit(X_polars.clone(), y_polars)
                else:
                    gators_transformer.fit(X_polars.clone())
                gators_transformer.transform(X_polars.clone())
        except TimeoutError:
            print("⚠️  Gators warmup timed out")
        except Exception:
            pass
        
        try:
            with timeout(timeout_seconds):
                if y_pandas is not None:
                    comparison_transformer.fit(X_pandas.copy(), y_pandas)
                else:
                    comparison_transformer.fit(X_pandas.copy())
                comparison_transformer.transform(X_pandas.copy())
        except TimeoutError:
            print("⚠️  Comparison warmup timed out")
        except Exception:
            pass
    
    # Benchmark Gators
    gators_fit_times = []
    gators_transform_times = []
    
    for run in range(n_runs):
        X_copy = X_polars.clone()
        
        try:
            with timeout(timeout_seconds):
                start = time.perf_counter()
                if y_polars is not None:
                    gators_transformer.fit(X_copy, y_polars)
                else:
                    gators_transformer.fit(X_copy)
                gators_fit_times.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                _ = gators_transformer.transform(X_copy)
                gators_transform_times.append(time.perf_counter() - start)
        except TimeoutError:
            print(f"⚠️  Gators run {run+1}/{n_runs} timed out after {timeout_seconds}s")
            raise
    
    results['gators_fit'] = np.median(gators_fit_times)
    results['gators_transform'] = np.median(gators_transform_times)
    results['gators_total'] = results['gators_fit'] + results['gators_transform']
    
    # Benchmark comparison library (feature-engine or sklearn)
    comparison_fit_times = []
    comparison_transform_times = []
    comparison_timed_out = False
    
    for run in range(n_runs):
        X_copy = X_pandas.copy()
        
        try:
            with timeout(timeout_seconds):
                start = time.perf_counter()
                if y_pandas is not None:
                    comparison_transformer.fit(X_copy, y_pandas)
                else:
                    comparison_transformer.fit(X_copy)
                comparison_fit_times.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                _ = comparison_transformer.transform(X_copy)
                comparison_transform_times.append(time.perf_counter() - start)
        except TimeoutError:
            print(f"⚠️  Comparison run {run+1}/{n_runs} timed out after {timeout_seconds}s - using timeout for speedup calculation")
            comparison_timed_out = True
            break  # Stop trying more runs
    
    # If comparison timed out, use timeout value as minimum time
    if comparison_timed_out:
        results['comparison_fit'] = timeout_seconds  # Minimum estimate
        results['comparison_transform'] = timeout_seconds  # Minimum estimate
        results['comparison_total'] = timeout_seconds
        results['timed_out'] = True
    else:
        results['comparison_fit'] = np.median(comparison_fit_times)
        results['comparison_transform'] = np.median(comparison_transform_times)
        results['comparison_total'] = results['comparison_fit'] + results['comparison_transform']
        results['timed_out'] = False
    
    # Calculate speedup (avoid division by zero)
    if results['gators_fit'] > 0:
        results['speedup_fit'] = results['comparison_fit'] / results['gators_fit']
    else:
        results['speedup_fit'] = float('inf')
    
    if results['gators_transform'] > 0:
        results['speedup_transform'] = results['comparison_transform'] / results['gators_transform']
    else:
        results['speedup_transform'] = float('inf')
    
    if results['gators_total'] > 0:
        # When comparison timed out, speedup is at least timeout/gators
        results['speedup_total'] = results['comparison_total'] / results['gators_total']
    else:
        results['speedup_total'] = float('inf')
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_benchmark_summary(
    dataset_size: int,
    results: Dict[str, float],
    transformer_name: str = "Transformer"
) -> None:
    """Print a formatted summary of benchmark results.
    
    Parameters
    ----------
    dataset_size : int
        Number of rows in the benchmarked dataset.
    results : Dict[str, float]
        Results dictionary from benchmark_transformer().
    transformer_name : str, default="Transformer"
        Name of the transformer being benchmarked.
    """
    print(f"  {dataset_size:>8,} rows: "
          f"Gators={results['gators_total']:.4f}s, "
          f"Comparison={results['comparison_total']:.4f}s, "
          f"Speedup={results['speedup_total']:.2f}x "
          f"({transformer_name})")


def validate_dataframes_equal(
    df_polars: pl.DataFrame,
    df_pandas: pd.DataFrame,
    tolerance: float = 1e-10
) -> bool:
    """Validate that Polars and pandas DataFrames contain equivalent data.
    
    Parameters
    ----------
    df_polars : pl.DataFrame
        Polars DataFrame to compare.
    df_pandas : pd.DataFrame
        pandas DataFrame to compare.
    tolerance : float, default=1e-10
        Tolerance for floating-point comparisons.
        
    Returns
    -------
    bool
        True if DataFrames are equivalent, False otherwise.
    """
    # Check shapes
    if df_polars.shape != df_pandas.shape:
        return False
    
    # Check column names
    if list(df_polars.columns) != list(df_pandas.columns):
        return False
    
    # Check data values (convert to numpy for comparison)
    polars_values = df_polars.to_numpy()
    pandas_values = df_pandas.values
    
    return np.allclose(polars_values, pandas_values, rtol=tolerance, atol=tolerance, equal_nan=True)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_speedup_scaling(
    all_results: pd.DataFrame,
    dataset_sizes: list,
    type_column: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    show_annotation: bool = True
) -> None:
    """Create the most illustrative speedup scaling plot for benchmark results.
    
    This plot shows how Gators' performance advantage scales with dataset size
    for each transformer type. It's the single most informative visualization
    for understanding:
    - Which transformers benefit most from Polars parallelization
    - How performance advantage changes with scale
    - Comparison across different strategies
    
    Parameters
    ----------
    all_results : pd.DataFrame
        Concatenated results DataFrame containing:
        - dataset_size: int
        - speedup_total: float
        - {type}_type: str (e.g., 'encoder_type', 'imputer_type', 'scaler_type')
    dataset_sizes : list
        List of dataset sizes used in benchmarks (e.g., [1000, 10000, 100000, 1000000]).
    type_column : Optional[str], default=None
        Name of the column containing transformer types. If None, auto-detects
        the first column ending with '_type'.
    figsize : Tuple[int, int], default=(14, 8)
        Figure size (width, height) in inches.
    show_annotation : bool, default=True
        Whether to annotate the peak speedup value.
        
    Returns
    -------
    None
        Displays the plot using matplotlib.
        
    Examples
    --------
    >>> # After concatenating all benchmark results
    >>> all_results = pd.concat([
    ...     onehot_results_df,
    ...     ordinal_results_df,
    ...     ...
    ... ], ignore_index=True)
    >>> plot_speedup_scaling(all_results, DATASET_SIZES)
    """
    # Auto-detect the type column if not provided
    if type_column is None:
        for col in all_results.columns:
            if col.endswith('_type'):
                type_column = col
                break
        
        if type_column is None:
            raise ValueError(
                "No column ending with '_type' found in all_results. "
                "Expected 'encoder_type', 'imputer_type', 'scaler_type', etc. "
                "Or specify type_column parameter explicitly."
            )
    
    # Extract transformer category (e.g., 'encoder' from 'encoder_type')
    category = type_column.replace('_type', '').capitalize()
    
    # Get unique transformer types
    transformer_types = sorted(all_results[type_column].unique())
    n_types = len(transformer_types)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dynamic color and marker selection
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_types))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h'][:n_types]
    
    # Plot each transformer type
    for idx, transformer_type in enumerate(transformer_types):
        data = all_results[all_results[type_column] == transformer_type]
        
        # Average speedup for each dataset size
        speedups = [data[data['dataset_size'] == size]['speedup_total'].mean() 
                    for size in dataset_sizes]
        
        ax.plot(dataset_sizes, speedups, 
                marker=markers[idx], 
                label=transformer_type, 
                linewidth=2.5, 
                markersize=10,
                color=colors[idx],
                alpha=0.85)
    
    # Axis configuration
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size (rows)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup (Gators vs Feature-engine)', fontsize=14, fontweight='bold')
    ax.set_title(f'{category} Performance Scaling: How Gators\' Advantage Grows with Data Size', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Reference line at 1x (no speedup)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.6, 
               label='No speedup (1x)', zorder=0)
    
    # Formatting
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, ncol=2 if n_types > 6 else 1)
    ax.set_xticks(dataset_sizes)
    ax.set_xticklabels([f'{size:,}' for size in dataset_sizes])
    
    # Annotate peak performance
    if show_annotation:
        max_speedup_row = all_results.loc[all_results['speedup_total'].idxmax()]
        max_type = max_speedup_row[type_column]
        ax.annotate(f"Peak: {max_speedup_row['speedup_total']:.1f}x\n({max_type})",
                    xy=(max_speedup_row['dataset_size'], max_speedup_row['speedup_total']),
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))
    
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print("\n📊 KEY INSIGHTS:")
    print("="*70)
    print("• Upward slopes → Polars advantage INCREASES with scale")
    print("• Higher lines → Better overall performance")
    print("• Log-log scale reveals exponential vs linear scaling patterns")
    
    # Best performer analysis
    avg_speedups = all_results.groupby(type_column)['speedup_total'].mean().sort_values(ascending=False)
    best_type = avg_speedups.idxmax()
    best_speedup = avg_speedups.max()
    worst_type = avg_speedups.idxmin()
    worst_speedup = avg_speedups.min()
    
    print(f"• Best average performer: {best_type} ({best_speedup:.1f}x faster)")
    print(f"• Most challenging: {worst_type} ({worst_speedup:.1f}x faster)")
    print(f"• Overall average: {all_results['speedup_total'].mean():.1f}x speedup")
    print(f"• Overall median: {all_results['speedup_total'].median():.1f}x speedup")
    print("="*70)
