import cython
import numpy as np
cimport numpy as np

ctypedef fused num_float_t:
    np.float32_t
    np.float64_t


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_float_t, ndim = 2] standard_scaler(
        np.ndarray[num_float_t, ndim=2] X,
        np.ndarray[num_float_t, ndim=1] X_mean,
        np.ndarray[num_float_t, ndim=1] X_std,

):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                X[i, j] = (X[i, j] - X_mean[j]) / X_std[j]
    return X

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_float_t, ndim = 2]  minmax_scaler(
        np.ndarray[num_float_t, ndim=2] X,
        np.ndarray[num_float_t, ndim=1] X_min,
        np.ndarray[num_float_t, ndim=1] X_max,

):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                X[i, j] = (X[i, j] - X_min[j]) / (X_max[j] - X_min[j])
    return X