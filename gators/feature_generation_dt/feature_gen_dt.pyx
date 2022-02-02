import cython
import numpy as np
cimport numpy as np
from libc.math cimport cos, isnan, pi, sin



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] ordinal_datetime(
        np.ndarray[object, ndim=2] X, np.ndarray[np.int64_t, ndim=1] bounds):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int start_idx = bounds[0]
    cdef int end_idx = bounds[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, j] = np.nan
            else:
                X_new[i, j] = float(str(X[i, j])[start_idx: end_idx])
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] ordinal_day_of_week(
        np.ndarray[object, ndim=2] X,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, j] = np.nan
            else:
                X_new[i, j] = (np.array(X[i, j]).astype('datetime64[D]').astype('float64') - 4) % 7

    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_minute_of_hour(
        np.ndarray[object, ndim=2] X,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * float(str(X[i, j])[14: 16])
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_hour_of_day(
        np.ndarray[object, ndim=2] X,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * float(str(X[i, j])[11: 13])
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_day_of_week(
        np.ndarray[object, ndim=2] X,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * float((np.array(X[i, j]).astype('datetime64[D]').astype('float64') - 4) % 7)
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_month_of_year(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] bounds,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int start_idx = bounds[0]
    cdef int end_idx = bounds[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                val = PREFACTOR * (float(str(X[i, j])[start_idx: end_idx]) -1.)
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] cyclic_day_of_month(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] bounds,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef int start_idx = bounds[0]
    cdef int end_idx = bounds[1]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef int days_in_month 

    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, j] is None) or (X[i, j] != X[i, j]):
                X_new[i, 2*j] = np.nan
                X_new[i, 2*j+1] = np.nan
            else:
                days_in_month = (((np.array(X[i, j]).astype('datetime64[M]')+1).astype('datetime64[D]') - np.array(X[i, j]).astype('datetime64[M]')) // np.timedelta64(1, 'D') - 1)
                val = 2 * pi * (float(str(X[i, j])[start_idx: end_idx]) -1.) / days_in_month
                X_new[i, 2*j] = cos(val)
                X_new[i, 2*j+1] = sin(val)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] deltatime(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_a,
        np.ndarray[np.int64_t, ndim=1] idx_columns_b,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, idx_columns_a[j]] == None) or (X[i, idx_columns_a[j]] != X[i, idx_columns_a[j]]) or (X[i, idx_columns_b[j]] == None) or (X[i, idx_columns_b[j]] != X[i, idx_columns_b[j]]):
                X_new[i, j] = np.nan
            else:
                X_new[i, j] = (np.array(X[i, idx_columns_a[j]]).astype(str).astype('<M8[s]') - np.array(X[i, idx_columns_b[j]]).astype(str).astype('<M8[s]')).astype(int).astype(float)
    return X_new
