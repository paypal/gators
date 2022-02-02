# License: Apache-2.0
cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] extract_str(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[np.int64_t, ndim=1] i_min_vec,
        np.ndarray[np.int64_t, ndim=1] i_max_vec,
        ):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t k
    cdef object value
    cdef np.int64_t n_rows = X.shape[0]
    cdef np.int64_t n_columns = idx_columns.shape[0]
    cdef np.ndarray X_new = np.empty((n_rows, n_columns), object)
    for k in range(n_rows):
        for i in range(n_columns):
            value = X[k, idx_columns[i]]
            if value is None or isinstance(value, float) or (i_max_vec[i] > len(value)):
                X_new[k, i] = ''
                continue
            X_new[k, i] = X[k, idx_columns[i]][i_min_vec[i]: i_max_vec[i]]
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] split_and_extract_str(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[object, ndim=1] str_split_vec,
        np.ndarray[np.int64_t, ndim=1] idx_split_vec):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t k
    cdef object value
    cdef list value_split
    cdef np.int64_t n_rows = X.shape[0]
    cdef np.int64_t n_columns = idx_columns.shape[0]
    cdef np.ndarray X_new = np.empty((n_rows, n_columns), object)
    for k in range(n_rows):
        for i in range(n_columns):
            value = X[k, idx_columns[i]]
            if value is None or isinstance(value, float):
                X_new[k, i] = 'MISSING'
                continue
            value_split = value.split(str_split_vec[i])
            if len(value_split) <= idx_split_vec[i]:
                X_new[k, i] = 'MISSING'
                continue
            X_new[k, i] = value_split[idx_split_vec[i]]
    return np.concatenate((X, X_new), axis=1)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] string_length(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.empty((n_rows, n_cols))
    cdef object val
    for j in range(n_cols):
        for i in range(n_rows):
            val = X[i, idx_columns[j]]
            if val is None or val == 'nan':
                X_new[i, j] = 0
                continue
            X_new[i, j] = len(str(val))
    return np.concatenate((X, X_new),axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] contains(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int_t, ndim=1] idx_columns,
        np.ndarray[object, ndim=1] contains_vec,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float_t, ndim=2] X_new = np.zeros((n_rows, n_cols))
    cdef object val
    for j in range(n_cols):
        for i in range(n_rows):
            val = contains_vec[j]
            if val in X[i, idx_columns[j]]:
                X_new[i, j] = 1.
    return np.concatenate((X, X_new),axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] upper_case(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    for j in range(n_cols):
        for i in range(n_rows):
            if X[i, idx_columns[j]] is None or X[i, idx_columns[j]] == 'nan':
                continue
            X[i, idx_columns[j]] = X[i, idx_columns[j]].upper()
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] lower_case(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    for j in range(n_cols):
        for i in range(n_rows):
            if X[i, idx_columns[j]] is None or X[i, idx_columns[j]] == 'nan':
                continue
            X[i, idx_columns[j]] = X[i, idx_columns[j]].lower()
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] isin(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[object, ndim=2] values_vec_np,
        np.ndarray[np.int64_t, ndim=1] n_values_vec

):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] X_new = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            for k in range(n_values_vec[j]):
                if X[i, idx_columns[j]] == values_vec_np[j, k]:
                    X_new[i, j] = 1.
                    break
    return np.concatenate((X, X_new), axis=1)