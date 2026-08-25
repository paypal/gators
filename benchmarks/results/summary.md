# Benchmark results
Generated on macOS-26.6.2-arm64-arm-64bit-Mach-O, Python 3.14.5.

### 50,000 rows

| Transformer | gators (s) | scikit-learn (s) | feature-engine (s) | speedup vs sklearn | speedup vs feature-engine |
|---|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.001 | 0.004 | 0.002 | 3.7x | 2.0x |
| StandardScaler | 0.001 | 0.001 | n/a | 1.6x | n/a |
| QuantileClipper | 0.001 | n/a | 0.009 | n/a | 10.1x |
| EqualSizeDiscretizer (5 bins) | 0.003 | 0.012 | 0.025 | 3.6x | 7.6x |
| OneHotEncoder | 0.006 | 0.027 | 0.051 | 4.5x | 8.4x |
| OrdinalEncoder | 0.004 | 0.026 | 0.015 | 5.9x | 3.4x |
| TargetEncoder | 0.006 | 0.045 | 0.022 | 7.3x | 3.5x |
| WOEEncoder | 0.005 | n/a | 0.022 | n/a | 4.2x |

### 500,000 rows

| Transformer | gators (s) | scikit-learn (s) | feature-engine (s) | speedup vs sklearn | speedup vs feature-engine |
|---|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.005 | 0.025 | 0.012 | 5.2x | 2.4x |
| StandardScaler | 0.002 | 0.009 | n/a | 3.8x | n/a |
| QuantileClipper | 0.005 | n/a | 0.056 | n/a | 12.3x |
| EqualSizeDiscretizer (5 bins) | 0.025 | 0.112 | 0.182 | 4.5x | 7.3x |
| OneHotEncoder | 0.045 | 0.283 | 0.320 | 6.2x | 7.0x |
| OrdinalEncoder | 0.029 | 0.267 | 0.127 | 9.1x | 4.3x |
| TargetEncoder | 0.029 | 0.442 | 0.163 | 15.0x | 5.5x |
| WOEEncoder | 0.028 | n/a | 0.181 | n/a | 6.5x |
