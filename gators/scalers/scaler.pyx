import cython
import numpy as np
cimport numpy as np

ctypedef fused num_float_t:
    np.float32_t
    np.float64_t

cdef extern from "math.h":
    double log(double x) nogil  


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_float_t, ndim = 2]  scaler(
        np.ndarray[num_float_t, ndim=2] X,
        np.ndarray[num_float_t, ndim=1] X_offset,
        np.ndarray[num_float_t, ndim=1] X_scale,

):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                X[i, j] = (X[i, j] - X_offset[j]) * X_scale[j]
    return X



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[num_float_t, ndim = 2]  yeo_johnson(
        np.ndarray[num_float_t, ndim=2] X,
        np.ndarray[num_float_t, ndim=1] lambdas,

):
    cdef int i
    cdef int j
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef float lmbda
    cdef float x
    with nogil:
        for i in range(n_rows):
            for j in range(n_cols):
                lmbda = lambdas[j]
                x = X[i, j]
                if lmbda == 0:
                    if X[i, j] >= 0:
                        X[i, j] = log(x+1)
                    else:
                        X[i, j] = - ((-x+1) ** (2-lmbda) - 1) / (2 - lmbda)
                elif lmbda == 2:
                    if X[i, j] >= 0:
                        X[i, j] = ((x+1) ** lmbda - 1) / lmbda
                    else:
                        X[i, j] = -log(-x+1)
                else:
                    if X[i, j] >= 0:
                        X[i, j] = ((x+1) ** lmbda - 1) / lmbda
                    else:
                        X[i, j] = -((-x+1) ** (2-lmbda) - 1) / (2 - lmbda)  
    return X