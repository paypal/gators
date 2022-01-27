# License: Apache-2.0
cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] encoder(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] num_categories_vec,
        np.ndarray[object, ndim=2] values_vec,
        np.ndarray[np.float64_t, ndim=2] encoded_values_vec,
        np.ndarray[np.int64_t, ndim=1] idx_columns):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t k
    cdef np.int64_t j_max
    cdef object value
    cdef np.int64_t n_rows = X.shape[0]
    cdef np.int64_t n_cols = idx_columns.shape[0]
    for k in range(n_rows):
        for i in range(n_cols):
            value = X[k, idx_columns[i]]
            j_max = num_categories_vec[i]
            for j in range(j_max):
                if value == values_vec[i, j]:
                    X[k, idx_columns[i]] = encoded_values_vec[i, j]
                    break
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] onehot_encoder(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[object, ndim=1] cats,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray X_new = np.empty(
        (n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = X[i, idx_columns[j]] == cats[j]
    return np.concatenate((X, X_new.astype(np.float64)), axis=1)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] binned_columns_encoder(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t n_rows = X.shape[0]
    cdef np.int64_t n_cols = idx_columns.shape[0]
    cdef np.ndarray X_new = np.empty(
        (n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = float(X[i, idx_columns[j]][1:])
    return np.concatenate((X, X_new), axis=1).astype(object)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim=2] binned_columns_encoder_inplace(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns):
    cdef np.int64_t i
    cdef np.int64_t j
    cdef np.int64_t n_rows = X.shape[0]
    cdef np.int64_t n_cols = idx_columns.shape[0]
    for i in range(n_rows):
        for j in range(n_cols):
            X[i, idx_columns[j]] = float(X[i, idx_columns[j]][1:])
    print(X.dtype)
    print(X.astype(object).dtype)
    return X.astype(object)
