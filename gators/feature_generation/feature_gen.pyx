import cython
import numpy as np

cimport numpy as np
from libc.math cimport cos, isnan, pi, sin, sqrt

ctypedef fused num_t:
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

ctypedef fused num_float_t:
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] one_hot(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[object, ndim=1] cats,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[object, ndim = 2] X_new = np.empty(
        (n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = X[i, idx_columns[j]] == cats[j]
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_t, ndim = 2] is_null(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[num_t, ndim = 2] X_new = np.empty((n_rows, n_cols))
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                X_new[i, j] = X[i, idx_columns[j]] != X[i, idx_columns[j]]
    return np.concatenate((X, X_new), axis=1)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] is_null_object(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] X_new = np.zeros(
        (n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            if (X[i, idx_columns[j]] is None) or (X[i, idx_columns[j]] != X[i, idx_columns[j]]):
                X_new[i, j] = 1.
    return np.concatenate((X, X_new), axis=1)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_t, ndim = 2] is_equal(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_a,
        np.ndarray[np.int64_t, ndim=1] idx_columns_b,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_a.shape[0]
    cdef np.ndarray[num_t, ndim = 2] X_new = np.empty((n_rows, n_cols), X.dtype)
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                X_new[i, j] = X[i, idx_columns_a[j]] == X[i, idx_columns_b[j]]
    return np.concatenate((X, X_new), axis=1)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] is_equal_object(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_a,
        np.ndarray[np.int64_t, ndim=1] idx_columns_b,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_a.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] X_new = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            X_new[i, j] = X[i, idx_columns_a[j]] == X[i, idx_columns_b[j]]
    return np.concatenate((X, X_new), axis=1)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim = 2] cluster_statistics(
        np.ndarray[num_float_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=2] idx_columns,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef int n_elements = idx_columns.shape[1]
    cdef np.ndarray[np.float64_t, ndim = 2] X_new = np.zeros((n_rows, 2 * n_cols))
    cdef num_float_t mean
    cdef num_float_t std
    cdef num_float_t denumerator = (n_elements - 1)
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                mean = 0
                std = 0
                for k in range(n_elements):
                    mean = mean + X[i, idx_columns[j, k]]
                mean /= n_elements
                for k in range(n_elements):
                    std += (X[i, idx_columns[j, k]] - mean) * \
                        (X[i, idx_columns[j, k]] - mean)
                X_new[i, 2 * j] = mean
                X_new[i, 2 * j + 1] = sqrt(std) / denumerator
    return X_new


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim = 2] elementary_arithmetics(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_a,
        np.ndarray[np.int64_t, ndim=1] idx_columns_b,
        object operator,
        float coef,
        float EPSILON,
):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_a.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] X_new = np.empty((n_rows, n_cols))
    if operator == '+':
        with nogil:
            for i in range(n_rows):
                for j in range(n_cols):
                    X_new[i, j] = X[i, idx_columns_a[j]] + coef * X[i, idx_columns_b[j]]
    elif operator == '*':
        with nogil:
            for i in range(n_rows):
                for j in range(n_cols):
                    X_new[i, j] = X[i, idx_columns_a[j]] * X[i, idx_columns_b[j]]
    else:
        with nogil:
            for i in range(n_rows):
                for j in range(n_cols):
                    X_new[i, j] = X[i, idx_columns_a[j]] / (X[i, idx_columns_b[j]] + EPSILON)
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_t, ndim = 2] plan_rotation(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns_x,
        np.ndarray[np.int64_t, ndim=1] idx_columns_y,
        np.ndarray[np.float64_t, ndim=1] cos_vec,
        np.ndarray[np.float64_t, ndim=1] sin_vec,
):
    cdef int i
    cdef int j
    cdef int k
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns_x.shape[0]
    cdef int n_elements = cos_vec.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] X_new = np.empty(
        (n_rows, 2 * n_cols * n_elements), np.float64)
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                for k in range(n_elements):
                    X_new[i, 2*k + 2*j*n_elements] = X[i, idx_columns_x[j]] * cos_vec[k] - X[i, idx_columns_y[j]] * sin_vec[k]
                    X_new[i, 2*k+1+ 2*j*n_elements] = X[i, idx_columns_x[j]] * sin_vec[k] + X[i, idx_columns_y[j]] * cos_vec[k]
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_t, ndim = 2] polynomial(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=2] combinations_np,
        int degree,
):
    cdef int n_rows = X.shape[0]
    cdef int n_cols = combinations_np.shape[0]
    cdef np.ndarray[num_t, ndim = 2] X_new = np.ones((n_rows, n_cols))
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                for k in range(degree):
                    if combinations_np[j, k] >= 0:
                        X_new[i, j] *= X[i, combinations_np[j, k]]
    return X_new


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[object, ndim = 2] polynomial_object(
        np.ndarray[object, ndim=2] X,
        np.ndarray[np.int64_t, ndim=2] combinations_np,
        int degree,
):
    cdef int n_rows = X.shape[0]
    cdef int n_cols = combinations_np.shape[0]
    cdef np.ndarray[object, ndim = 2] X_new = np.zeros((n_rows, n_cols), object)
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef object val
    for i in range(n_rows):
        for j in range(n_cols):
            val = ''
            for k in range(degree):
                if combinations_np[j, k] >= 0:
                    if X[i, combinations_np[j, k]] is None:
                        continue
                    val += str(X[i, combinations_np[j, k]])
            X_new[i, j] = val
    return np.concatenate((X, X_new), axis=1)