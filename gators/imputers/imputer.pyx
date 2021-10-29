# License: Apache-2.0
cimport cython
cimport numpy as np
from libc.math cimport isnan

ctypedef fused num_float_t:
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] object_imputer(
        np.ndarray[object, ndim=2] X,
        np.ndarray[object, ndim=1] statistics,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    """Object imputer.

    Args:
        X (np.ndarray): Input array.
        statistics(np.ndarray): Imputation values.
        idx_columns (np.ndarray): Array of column indices.
   
    Returns:
        np.ndarray: Imputed array.
    """
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_idx_columns = idx_columns.shape[0]
    for j in range(n_rows):
        for i in range(n_idx_columns):
            if (X[j, idx_columns[i]] != X[j, idx_columns[i]]) or (X[j, idx_columns[i]] is None):
                X[j, idx_columns[i]] = statistics[i]
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_float_t, ndim = 2] float_imputer(
        np.ndarray[num_float_t, ndim=2] X,
        np.ndarray[np.float64_t, ndim=1] statistics,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    """Float imputer.

    Args:
        X (np.ndarray): Input array.
        statistics (np.ndarray): Imputation values.
        idx_columns (np.ndarray): Array of column indices.
   
    Returns:
        np.ndarray: Imputed array.
    """

    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_idx_columns = idx_columns.shape[0]
    with nogil:
        for j in range(n_rows):
            for i in range(n_idx_columns):
                if isnan(X[j, idx_columns[i]]):
                    X[j, idx_columns[i]] = statistics[i]
    return X


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] float_imputer_object(
        np.ndarray[object, ndim=2] X,
        np.ndarray[object, ndim=1] statistics,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    """Float imputer.

    Args:
        X (np.ndarray): Input array.
        statistics(np.ndarray): Imputation values.
        idx_columns (np.ndarray): Array of column indices.
   
    Returns:
        np.ndarray: Imputed array.
    """

    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_idx_columns = idx_columns.shape[0]
    for j in range(n_rows):
        for i in range(n_idx_columns):
            if isnan(X[j, idx_columns[i]]):
                X[j, idx_columns[i]] = statistics[i]
    return X

