"""
GENERIC ILLUSTRATIVE PLOT for ANY Benchmark Results
====================================================

This script creates the most illustrative visualization for benchmark comparisons
between Gators (Polars) and Feature-engine (pandas) transformers.

REQUIREMENTS:
- all_results: pandas DataFrame with columns:
  * dataset_size: int (e.g., 1000, 10000, 100000, 1000000)
  * speedup_total: float (Gators/Feature-engine performance ratio)
  * {type}_type: str (e.g., 'encoder_type', 'imputer_type', 'scaler_type')
    - Automatically detects the type column ending with '_type'
  
- dataset_sizes: list of dataset sizes used in benchmarks

USAGE:
1. After concatenating all result DataFrames into all_results
2. Run this cell to generate the speedup scaling plot
3. Works for encoders, imputers, scalers, datetime features, etc.

WHY THIS IS THE MOST ILLUSTRATIVE PLOT:
✓ Shows ALL transformer types in one view
✓ Reveals scalability trends (does speedup increase with data size?)
✓ Easy comparison across strategies
✓ Log-log scale reveals scaling patterns (linear, exponential, plateau)
✓ Identifies which transformers benefit most from Polars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Auto-detect the type column (encoder_type, imputer_type, etc.)
# ============================================================================
type_column = None
for col in all_results.columns:
    if col.endswith('_type'):
        type_column = col
        break

if type_column is None:
    raise ValueError("No column ending with '_type' found in all_results. "
                     "Expected 'encoder_type', 'imputer_type', 'scaler_type', etc.")

# Extract transformer category (e.g., 'encoder' from 'encoder_type')
category = type_column.replace('_type', '').capitalize()

# Get unique transformer types
transformer_types = sorted(all_results[type_column].unique())
n_types = len(transformer_types)

# ============================================================================
# MOST ILLUSTRATIVE PLOT: Speedup Scaling Across Dataset Sizes
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

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

# ============================================================================
# Print key insights
# ============================================================================
print("\n📊 KEY INSIGHTS FROM THIS PLOT:")
print("="*70)
print("• Upward slopes → Polars advantage INCREASES with scale (production gold!)")
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
