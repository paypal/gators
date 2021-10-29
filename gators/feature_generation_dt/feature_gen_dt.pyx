import cython
import numpy as np

cimport numpy as np
from libc.math cimport cos, isnan, pi, sin


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] ordinal_minute_of_hour(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[object, ndim=2] X_new = np.empty((n_rows, n_cols), object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = str(float(X[i, idx_columns[j]].minute))
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] ordinal_hour_of_day(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[object, ndim=2] X_new = np.empty((n_rows, n_cols), object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = str(float(X[i, idx_columns[j]].hour))
    return np.concatenate((X, X_new), axis=1)

   
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] ordinal_day_of_week(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[object, ndim=2] X_new = np.empty((n_rows, n_cols), object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = str(float(X[i, idx_columns[j]].dayofweek))
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] ordinal_day_of_month(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[object, ndim=2] X_new = np.empty((n_rows, n_cols),object)
    cdef object val
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = str(float(X[i, idx_columns[j]].day))
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] ordinal_month_of_year(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[object, ndim=2] X_new = np.empty((n_rows, n_cols), object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = str(float(X[i, idx_columns[j]].month))
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] cyclic_minute_of_hour(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            val = PREFACTOR * X[i, idx_columns[j]].minute
            X_new[i, 2*j] = cos(val)
            X_new[i, 2*j+1] = sin(val)
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] cyclic_hour_of_day(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            val = PREFACTOR * X[i, idx_columns[j]].hour
            X_new[i, 2*j] = cos(val)
            X_new[i, 2*j+1] = sin(val)
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] cyclic_day_of_week(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            val = PREFACTOR * X[i, idx_columns[j]].dayofweek
            X_new[i, 2*j] = cos(val)
            X_new[i, 2*j+1] = sin(val)
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] cyclic_month_of_year(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        double PREFACTOR,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            val = PREFACTOR * (X[i, idx_columns[j]].month - 1.)
            X_new[i, 2*j] = cos(val)
            X_new[i, 2*j+1] = sin(val)
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] cyclic_day_of_month(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, 2 * n_cols))
    cdef double val
    for i in range(n_rows):
        for j in range(n_cols):
            val = 2 * pi *( X[i, idx_columns[j]].day - 1.) / (X[i, idx_columns[j]].daysinmonth - 1.)
            X_new[i, 2*j] = cos(val)
            X_new[i, 2*j+1] = sin(val)
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] deltatime(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_a,
        np.ndarray[np.int64_t, ndim=1] idx_columns_b,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    cdef object val_a
    cdef object val_b
    for i in range(n_rows):
        for j in range(n_cols):
            val_a = X[i, idx_columns_a[j]]
            val_b = X[i, idx_columns_b[j]]
            X_new[i, j] = (val_a - val_b).total_seconds()
    return np.concatenate((X, X_new), axis=1)
