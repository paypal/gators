# ONNX serving benchmark results
Generated on macOS-26.6.2-arm64-arm-64bit-Mach-O, Python 3.14.5.

`gators` 1.3.1, `polars` 1.43.0, `onnx` 1.22.0, `onnxruntime` 1.26.0.

Fit rows: 50,000. Serving rows sampled independently (seed=1) up to 100,000; each batch size is the first N rows of that serving set.

All parity checks (ONNX Runtime vs. native Polars `transform`, atol=0.0001) passed before any timing below was recorded.

### impute_scale

Serialized ONNX graph size: 6,135 bytes.

| Batch size | gators (s) | onnx (s) | onnx vs gators | gators (rows/s) | onnx (rows/s) | gators Δpeak-RSS (MB) | onnx Δpeak-RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000502 | 0.000322 | 1.56x | 1,991 | 3,102 | 0.5 | 0.0 |
| 10 | 0.000573 | 0.000317 | 1.81x | 17,443 | 31,550 | 0.1 | 0.0 |
| 100 | 0.000589 | 0.000397 | 1.48x | 169,815 | 251,783 | 0.2 | 0.0 |
| 1,000 | 0.000583 | 0.001055 | 0.55x | 1,715,142 | 947,867 | 0.4 | 0.0 |
| 10,000 | 0.000565 | 0.006680 | 0.08x | 17,710,870 | 1,496,997 | 2.8 | 1.4 |
| 100,000 | 0.000856 | 0.058543 | 0.01x | 116,816,698 | 1,708,153 | 34.6 | 51.6 |

### impute_clip_discretize

Serialized ONNX graph size: 19,938 bytes.

| Batch size | gators (s) | onnx (s) | onnx vs gators | gators (rows/s) | onnx (rows/s) | gators Δpeak-RSS (MB) | onnx Δpeak-RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000864 | 0.000308 | 2.81x | 1,158 | 3,249 | 0.0 | 0.0 |
| 10 | 0.000810 | 0.000308 | 2.63x | 12,349 | 32,494 | 0.0 | 0.0 |
| 100 | 0.000650 | 0.000365 | 1.78x | 153,955 | 273,941 | 0.0 | 0.0 |
| 1,000 | 0.000780 | 0.001182 | 0.66x | 1,281,366 | 846,024 | 0.0 | 0.0 |
| 10,000 | 0.001065 | 0.009345 | 0.11x | 9,393,711 | 1,070,058 | 0.0 | 0.0 |
| 100,000 | 0.003814 | 0.090926 | 0.04x | 26,221,193 | 1,099,794 | 12.3 | 106.1 |

### impute_encode_scale

Serialized ONNX graph size: 27,709 bytes.

| Batch size | gators (s) | onnx (s) | onnx vs gators | gators (rows/s) | onnx (rows/s) | gators Δpeak-RSS (MB) | onnx Δpeak-RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002486 | 0.000563 | 4.41x | 402 | 1,775 | 0.0 | 0.0 |
| 10 | 0.002640 | 0.000564 | 4.68x | 3,788 | 17,721 | 0.0 | 0.0 |
| 100 | 0.002577 | 0.000630 | 4.09x | 38,799 | 158,636 | 0.0 | 0.0 |
| 1,000 | 0.002588 | 0.001416 | 1.83x | 386,430 | 706,318 | 0.0 | 0.0 |
| 10,000 | 0.002625 | 0.009987 | 0.26x | 3,809,644 | 1,001,318 | 0.0 | 0.0 |
| 100,000 | 0.004792 | 0.096898 | 0.05x | 20,867,386 | 1,032,008 | 18.0 | 8.2 |
