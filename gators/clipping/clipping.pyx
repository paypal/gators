import cython
import numpy as np
cimport numpy as np

ctypedef fused num_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_t, ndim=2] clipping(
        np.ndarray[num_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=1] idx_columns,
        np.ndarray[num_t, ndim=2] clip_np,
):
    cdef int i
    cdef int j
    cdef int k
    cdef int n_rows = X.shape[0]
    cdef int n_cols = idx_columns.shape[0]
    cdef num_t val

    with nogil:
        for j in range(n_cols):
            for i in range(n_rows):
                val = X[i, j]
                if val < clip_np[j, 0]:
                    X[i, j] = clip_np[j, 0]
                    continue
                if val > clip_np[j, 1]:
                    X[i, j] = clip_np[j, 1]
                    continue
    return X