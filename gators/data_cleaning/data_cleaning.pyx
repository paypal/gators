
# License: Apache-2.0
cimport cython
cimport numpy as np

import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] replace(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[object, ndim=2] to_replace_np_keys,
        np.ndarray[object, ndim=2] to_replace_np_vals,
        np.ndarray[np.int64_t, ndim=1] n_elements_vec,
):
    cdef int i
    cdef int j
    cdef int k
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    for i in range(n_rows):
        for j in range(n_cols):
            for k in range(n_elements_vec[j]):
                if X[i, idx_columns[j]] == to_replace_np_keys[j, k]:
                    X[i, idx_columns[j]] = to_replace_np_vals[j, k]
                    break
    return X