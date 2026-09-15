# Extended benchmark matrix
Generated on macOS-26.6.2-arm64-arm-64bit-Mach-O, Python 3.14.5.

One-factor-at-a-time sweeps around a fixed baseline (8 numeric + 4 categorical columns, cardinality=20, null_frac=0.1); see benchmarks/README.md for why this is OFAT rather than a full factorial grid, and for the row/feature range actually covered versus the original proposal.

## Rows sweep

| transformer | gators_s | sklearn_s | feature_engine_s | speedup_vs_sklearn | speedup_vs_feature_engine | n_rows |
|---:|---:|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.0007 | 0.0011 | 0.0010 | 1.6603 | 1.5496 | 10000 |
| StandardScaler | 0.0006 | 0.0007 | n/a | 1.1436 | n/a | 10000 |
| QuantileClipper | 0.0006 | n/a | 0.0044 | n/a | 7.0721 | 10000 |
| EqualSizeDiscretizer (5 bins) | 0.0014 | 0.0028 | 0.0086 | 2.0410 | 6.2626 | 10000 |
| OneHotEncoder | 0.0024 | 0.0057 | 0.0232 | 2.3882 | 9.6459 | 10000 |
| OrdinalEncoder | 0.0016 | 0.0055 | 0.0039 | 3.4591 | 2.4447 | 10000 |
| TargetEncoder | 0.0018 | 0.0096 | 0.0071 | 5.2417 | 3.8817 | 10000 |
| WOEEncoder | 0.0027 | n/a | 0.0066 | n/a | 2.4264 | 10000 |
| NumericImputer (mean) | 0.0052 | 0.0457 | 0.0225 | 8.8261 | 4.3381 | 1000000 |
| StandardScaler | 0.0052 | 0.0174 | n/a | 3.3554 | n/a | 1000000 |
| QuantileClipper | 0.0081 | n/a | 0.1053 | n/a | 12.9501 | 1000000 |
| EqualSizeDiscretizer (5 bins) | 0.0492 | 0.2163 | 0.3418 | 4.3941 | 6.9447 | 1000000 |
| OneHotEncoder | 0.0766 | 0.5416 | 0.5558 | 7.0664 | 7.2523 | 1000000 |
| OrdinalEncoder | 0.0529 | 0.5022 | 0.2461 | 9.4989 | 4.6543 | 1000000 |
| TargetEncoder | 0.0466 | 0.8815 | 0.2949 | 18.9090 | 6.3271 | 1000000 |
| WOEEncoder | 0.0494 | n/a | 0.3057 | n/a | 6.1866 | 1000000 |
| NumericImputer (mean) | 0.0010 | 0.0035 | 0.0020 | 3.6525 | 2.0407 | 50000 |
| StandardScaler | 0.0008 | 0.0013 | n/a | 1.5884 | n/a | 50000 |
| QuantileClipper | 0.0009 | n/a | 0.0087 | n/a | 10.0870 | 50000 |
| EqualSizeDiscretizer (5 bins) | 0.0032 | 0.0117 | 0.0245 | 3.6162 | 7.5565 | 50000 |
| OneHotEncoder | 0.0060 | 0.0269 | 0.0508 | 4.4616 | 8.4299 | 50000 |
| OrdinalEncoder | 0.0044 | 0.0258 | 0.0151 | 5.8784 | 3.4274 | 50000 |
| TargetEncoder | 0.0062 | 0.0453 | 0.0217 | 7.2619 | 3.4835 | 50000 |
| WOEEncoder | 0.0051 | n/a | 0.0216 | n/a | 4.2105 | 50000 |
| NumericImputer (mean) | 0.0049 | 0.0250 | 0.0115 | 5.1514 | 2.3721 | 500000 |
| StandardScaler | 0.0024 | 0.0091 | n/a | 3.7767 | n/a | 500000 |
| QuantileClipper | 0.0046 | n/a | 0.0565 | n/a | 12.2867 | 500000 |
| EqualSizeDiscretizer (5 bins) | 0.0248 | 0.1117 | 0.1815 | 4.4956 | 7.3057 | 500000 |
| OneHotEncoder | 0.0455 | 0.2835 | 0.3199 | 6.2373 | 7.0378 | 500000 |
| OrdinalEncoder | 0.0294 | 0.2673 | 0.1266 | 9.0895 | 4.3034 | 500000 |
| TargetEncoder | 0.0295 | 0.4415 | 0.1627 | 14.9903 | 5.5237 | 500000 |
| WOEEncoder | 0.0276 | n/a | 0.1809 | n/a | 6.5479 | 500000 |

## Features sweep

| transformer | gators_s | sklearn_s | feature_engine_s | speedup_vs_sklearn | speedup_vs_feature_engine | n_numeric | n_categorical | n_features |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.0007 | 0.0048 | 0.0025 | 6.9794 | 3.7272 | 4 | 2 | 6 |
| StandardScaler | 0.0006 | 0.0020 | n/a | 3.1382 | n/a | 4 | 2 | 6 |
| QuantileClipper | 0.0011 | n/a | 0.0141 | n/a | 12.5280 | 4 | 2 | 6 |
| EqualSizeDiscretizer (5 bins) | 0.0082 | 0.0223 | 0.0378 | 2.7171 | 4.6068 | 4 | 2 | 6 |
| OneHotEncoder | 0.0076 | 0.0526 | 0.0602 | 6.8867 | 7.8908 | 4 | 2 | 6 |
| OrdinalEncoder | 0.0057 | 0.0493 | 0.0251 | 8.6222 | 4.3819 | 4 | 2 | 6 |
| TargetEncoder | 0.0053 | 0.0853 | 0.0309 | 16.1179 | 5.8419 | 4 | 2 | 6 |
| WOEEncoder | 0.0060 | n/a | 0.0321 | n/a | 5.3907 | 4 | 2 | 6 |
| NumericImputer (mean) | 0.0013 | 0.0095 | 0.0048 | 7.0770 | 3.5977 | 8 | 4 | 12 |
| StandardScaler | 0.0010 | 0.0035 | n/a | 3.6144 | n/a | 8 | 4 | 12 |
| QuantileClipper | 0.0017 | n/a | 0.0257 | n/a | 14.8246 | 8 | 4 | 12 |
| EqualSizeDiscretizer (5 bins) | 0.0108 | 0.0448 | 0.0767 | 4.1527 | 7.1075 | 8 | 4 | 12 |
| OneHotEncoder | 0.0148 | 0.1027 | 0.1224 | 6.9311 | 8.2634 | 8 | 4 | 12 |
| OrdinalEncoder | 0.0105 | 0.0973 | 0.0502 | 9.3045 | 4.8036 | 8 | 4 | 12 |
| TargetEncoder | 0.0102 | 0.1715 | 0.0620 | 16.7634 | 6.0563 | 8 | 4 | 12 |
| WOEEncoder | 0.0115 | n/a | 0.0624 | n/a | 5.4411 | 8 | 4 | 12 |
| NumericImputer (mean) | 0.0054 | 0.0442 | 0.0224 | 8.2236 | 4.1655 | 40 | 20 | 60 |
| StandardScaler | 0.0042 | 0.0169 | n/a | 4.0298 | n/a | 40 | 20 | 60 |
| QuantileClipper | 0.0056 | n/a | 0.1150 | n/a | 20.6063 | 40 | 20 | 60 |
| EqualSizeDiscretizer (5 bins) | 0.0322 | 0.2143 | 0.3597 | 6.6537 | 11.1695 | 40 | 20 | 60 |
| OneHotEncoder | 0.0701 | 0.5317 | 0.7646 | 7.5817 | 10.9020 | 40 | 20 | 60 |
| OrdinalEncoder | 0.0489 | 0.5121 | 0.2414 | 10.4623 | 4.9325 | 40 | 20 | 60 |
| TargetEncoder | 0.0466 | 0.9428 | 0.3039 | 20.2192 | 6.5169 | 40 | 20 | 60 |
| WOEEncoder | 0.0504 | n/a | 0.3251 | n/a | 6.4477 | 40 | 20 | 60 |
| NumericImputer (mean) | 0.0084 | 0.0680 | 0.0339 | 8.0555 | 4.0143 | 60 | 20 | 80 |
| StandardScaler | 0.0066 | 0.0253 | n/a | 3.8456 | n/a | 60 | 20 | 80 |
| QuantileClipper | 0.0081 | n/a | 0.1710 | n/a | 21.0487 | 60 | 20 | 80 |
| EqualSizeDiscretizer (5 bins) | 0.0476 | 0.3200 | 0.5387 | 6.7285 | 11.3276 | 60 | 20 | 80 |
| OneHotEncoder | 0.0665 | 0.5326 | 0.7173 | 8.0081 | 10.7856 | 60 | 20 | 80 |
| OrdinalEncoder | 0.0476 | 0.5218 | 0.2403 | 10.9657 | 5.0494 | 60 | 20 | 80 |
| TargetEncoder | 0.0476 | 0.9373 | 0.3002 | 19.6757 | 6.3027 | 60 | 20 | 80 |
| WOEEncoder | 0.0487 | n/a | 0.3232 | n/a | 6.6346 | 60 | 20 | 80 |

## Missingness sweep

| transformer | gators_s | sklearn_s | feature_engine_s | speedup_vs_sklearn | speedup_vs_feature_engine | null_frac |
|---:|---:|---:|---:|---:|---:|---:|
| NumericImputer (mean) | 0.0010 | 0.0059 | 0.0019 | 5.9794 | 1.9715 | 0.0000 |
| NumericImputer (mean) | 0.0013 | 0.0069 | 0.0039 | 5.4822 | 3.0949 | 0.0100 |
| NumericImputer (mean) | 0.0011 | 0.0094 | 0.0047 | 8.2931 | 4.1694 | 0.1000 |
| NumericImputer (mean) | 0.0013 | 0.0200 | 0.0081 | 15.5590 | 6.2965 | 0.5000 |

## Cardinality sweep

| transformer | gators_s | sklearn_s | feature_engine_s | speedup_vs_sklearn | speedup_vs_feature_engine | cardinality |
|---:|---:|---:|---:|---:|---:|---:|
| OneHotEncoder | 0.0014 | 0.0054 | 0.0032 | 3.7758 | 2.2438 | 5 |
| OrdinalEncoder | 0.0014 | 0.0050 | 0.0035 | 3.6622 | 2.5193 | 5 |
| TargetEncoder | 0.0021 | 0.0103 | 0.0050 | 4.9936 | 2.4198 | 5 |
| WOEEncoder | 0.0027 | n/a | 0.0051 | n/a | 1.8712 | 5 |
| OneHotEncoder | 0.0022 | 0.0066 | 0.0102 | 2.9895 | 4.6600 | 20 |
| OrdinalEncoder | 0.0014 | 0.0062 | 0.0040 | 4.5857 | 2.9486 | 20 |
| TargetEncoder | 0.0018 | 0.0127 | 0.0054 | 6.9275 | 2.9400 | 20 |
| WOEEncoder | 0.0023 | n/a | 0.0056 | n/a | 2.4606 | 20 |
| OneHotEncoder | 0.0031 | 0.0063 | 0.0439 | 2.0285 | 14.1772 | 100 |
| OrdinalEncoder | 0.0015 | 0.0055 | 0.0034 | 3.7008 | 2.3154 | 100 |
| TargetEncoder | 0.0019 | 0.0105 | 0.0050 | 5.4800 | 2.5841 | 100 |
| WOEEncoder | 0.0030 | n/a | 0.0051 | n/a | 1.7021 | 100 |
| OneHotEncoder | 0.0115 | 0.0115 | 1.3730 | 1.0043 | 119.8357 | 1000 |
| OrdinalEncoder | 0.0025 | 0.0079 | 0.0044 | 3.1105 | 1.7404 | 1000 |
| TargetEncoder | 0.0030 | 0.0147 | 0.0067 | 4.9072 | 2.2412 | 1000 |
| WOEEncoder | 0.0037 | n/a | 0.0066 | n/a | 1.7601 | 1000 |

## Threads sweep

| n_rows | transformer | threads | gators_s | sklearn_s | feature_engine_s |
|---:|---:|---:|---:|---:|---:|
| 50000 | NumericImputer (mean) | 1 | 0.0008 | 0.0035 | 0.0020 |
| 50000 | NumericImputer (mean) | default | 0.0007 | 0.0035 | 0.0020 |
| 50000 | OneHotEncoder | 1 | 0.0056 | 0.0269 | 0.0508 |
| 50000 | OneHotEncoder | default | 0.0047 | 0.0269 | 0.0508 |
| 50000 | TargetEncoder | 1 | 0.0088 | 0.0453 | 0.0217 |
| 50000 | TargetEncoder | default | 0.0033 | 0.0453 | 0.0217 |
| 500000 | NumericImputer (mean) | 1 | 0.0058 | 0.0250 | 0.0115 |
| 500000 | NumericImputer (mean) | default | 0.0027 | 0.0250 | 0.0115 |
| 500000 | OneHotEncoder | 1 | 0.0676 | 0.2835 | 0.3199 |
| 500000 | OneHotEncoder | default | 0.0355 | 0.2835 | 0.3199 |
| 500000 | TargetEncoder | 1 | 0.0929 | 0.4415 | 0.1627 |
| 500000 | TargetEncoder | default | 0.0237 | 0.4415 | 0.1627 |
