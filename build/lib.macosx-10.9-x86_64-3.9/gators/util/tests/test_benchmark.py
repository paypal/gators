# # License: Apache-2.0
# import pytest
# import numpy as np
# import pandas as pd
# from gators.scalers.minmax_scaler import MinMaxScaler
# from gators.scalers.standard_scaler import StandardScaler
# from gators.util import benchmark


# def test_get_runtime_in_milliseconds():
#     with pytest.raises(ValueError):
#         benchmark.get_runtime_in_milliseconds('5X  ± ')
#     assert benchmark.get_runtime_in_milliseconds('2 ns  ± ') == 2e-6
#     assert benchmark.get_runtime_in_milliseconds('3 µs  ± ') == 3e-3
#     assert benchmark.get_runtime_in_milliseconds('4 ms  ± ') == 4
#     assert benchmark.get_runtime_in_milliseconds('5 s  ± ') == 5e3


# def test_generate_per_sample_benchmarking():
#     data = np.arange(9).reshape(3, 3)
#     Xs = [
#         pd.DataFrame(
#             data, columns=list('ABC'), dtype=float),
#         pd.DataFrame(
#             data, columns=list('ABC'), dtype=float),
#     ]
#     objs = [
#         MinMaxScaler(),
#         StandardScaler(),
#     ]
#     extra_info_X_vec = ['Int', 'Float']
#     timeit_args = '-n 1 -r 1'

#     bench = benchmark.generate_per_sample_benchmarking(
#         Xs=Xs,
#         objs=objs,
#         extra_info_X_vec=extra_info_X_vec,
#         timeit_args=timeit_args
#     )
#     assert list(bench.columns) == ['pandas', 'numpy']
#     assert list(bench.index) == [
#         'MinMaxScalerInt', 'StandardScalerFloat',
#         'MinMaxScalerFloat', 'StandardScalerInt']
