# License: Apache-2.0
cimport cython
cimport numpy as np

import numpy as np

ctypedef fused num_t:
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def bin_rare_events(
        np.ndarray[object, ndim=2] X,
        np.ndarray[object, ndim=2] categories_to_keep_np,
        np.ndarray[np.int_t, ndim=1] n_categories_to_keep,
        np.ndarray[np.int_t, ndim=1] idx_columns):
    cdef np.int_t l
    cdef np.int_t j_col
    cdef np.int_t l_max
    cdef np.int_t n_rows = X.shape[0]
    cdef np.int_t n_cols = idx_columns.shape[0]
    cdef object val = 'OTHERS'
    cdef np.int_t is_rare = 1
    for j in range(n_cols):
        j_col = idx_columns[j]
        l_max = n_categories_to_keep[j]
        if l_max == 0:
            for k in range(n_rows):
                X[k, j_col] = val
        else:
            for k in range(n_rows):
                is_rare = 1
                for l in range(l_max):
                    if X[k, j_col] == categories_to_keep_np[l, j]:
                        is_rare = 0
                        break
                if is_rare:
                    X[k, j_col] = val
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_t, ndim = 2] bin_numerics(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[num_t, ndim=2] bin_limits,
        np.ndarray[num_t, ndim=2] bins,
        np.ndarray[np.int_t, ndim=1] idx_columns):
    cdef int i
    cdef int j
    cdef int k
    cdef int j_col
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef int n_bins = bin_limits.shape[0]
    cdef num_t val
    cdef np.ndarray[num_t, ndim=2] X_bin = np.empty((n_rows, n_cols), X.dtype)
    for i in range(n_rows):
        for j in range(n_cols):
            j_col = idx_columns[j]
            val = X[i, j_col]
            for k in range(1, n_bins):
                if val <= bin_limits[k, j]:
                    X_bin[i, j] = bins[k-1, j]
                    break
    return np.concatenate((X, X_bin), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] binning_inplace(
        np.ndarray[object, ndim=2] X,
        np.ndarray[num_t, ndim=2] bins_np,
        np.ndarray[np.int64_t, ndim=1] idx_columns):
    cdef int i
    cdef int j
    cdef int k
    cdef int j_col
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef int n_bins = bins_np.shape[0]
    for i in range(n_rows):
        for j in range(n_cols):
            j_col = idx_columns[j]
            for k in range(1, n_bins):
                if X[i, j_col] <= bins_np[k, j]:
                    X[i, j_col] = '_' + str(k - 1)
                    break
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] binning(
        np.ndarray[object, ndim=2] X,
        np.ndarray[num_t, ndim=2] bins_np,
        np.ndarray[np.int64_t, ndim=1] idx_columns):
    cdef int i
    cdef int j
    cdef int k
    cdef int j_col
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef int n_bins = bins_np.shape[0]
    cdef np.ndarray[object, ndim=2] X_bin = np.empty((n_rows, n_cols), object)
    for i in range(n_rows):
        for j in range(n_cols):
            j_col = idx_columns[j]
            for k in range(1, n_bins):
                if float(X[i, j_col]) <= bins_np[k, j]:
                    X_bin[i, j] = '_' + str(k - 1)
                    break
    return np.concatenate((X, X_bin), axis=1)