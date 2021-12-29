# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import numpy as np
from scipy.optimize import curve_fit
import scipy.sparse
import scipy.sparse.linalg as slinalg
import tqdm

cimport cython

# Import C++11 high-performance random number generators
cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 rng)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(T a, T b)
        T operator()(mt19937 rng)

# Compute ||v1 - v2||2^2
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double square_euclidean_dist(double[:] v1, double[:] v2) nogil:
    cdef double result = 0.0
    cdef double diff
    cdef Py_ssize_t vlen = len(v1), i
    for i in range(0, vlen, 1):
        diff = v1[i] - v2[i]
        result += diff * diff
    return result

# Clip overly large gradients to range [-4, 4]
cdef inline double fast_clip(double x) nogil:
    if x < -4.0:
        return -4.0
    if x > 4.0:
        return 4.0
    return x

# Gradients are computed with gradient_coefficient * (x - y)
cdef inline double grad_coeff_phi(double square_dist, double a, double b) nogil:
    cdef double neg_two_ab = -2.0 * a * b
    cdef double sd_pbm1 = square_dist ** (b-1)
    cdef double sd_pb = square_dist ** b
    return neg_two_ab * sd_pbm1 / (1.0 + a * sd_pb)

cdef inline double grad_coeff_neg_phi(double square_dist, double a, double b) nogil:
    # EPS = 1E-4
    cdef double two_b = 2.0 * b
    cdef double adj_sd = 1E-4 + square_dist
    cdef double sd_pb = square_dist ** b
    return two_b / (adj_sd * (1.0 + a * sd_pb))

def spectral_embedding(weights, d):
    # Sum rows(/columns) to obtain the degree matrix
    id_mat = scipy.sparse.identity(weights.shape[0])
    degree_diag = np.array( np.sum(weights, axis=0) )[0]
    degree_mat_sqrt_inv = scipy.sparse.diags(1.0 / np.sqrt(degree_diag))

    laplacian_mat = id_mat - degree_mat_sqrt_inv * weights * degree_mat_sqrt_inv
    ldim = laplacian_mat.shape[0]
    k = d + 1
    num_lanczos_vectors = max(2*k+1, int(np.sqrt(ldim)))
    lleigvals, lleigvecs = slinalg.eigen.eigsh(laplacian_mat,
                                               k=k,
                                               ncv=num_lanczos_vectors,
                                               which='SM',
                                               tol=1e-4,
                                               v0=np.ones(laplacian_mat.shape[0]),
                                               maxiter=ldim * 5)

    # Select the d SMALLEST eigen vectors EXCEPT the first one
    # Note that eigen vectors are in *columns*, not in rows
    return lleigvecs[:, 1:d+1]

# Optimize embedding for dimension 2
# This function is (hopefully) easy to optimize (with SIMD etc)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optimize_embedding2(weights, double[:, :] Y, min_dist, spread, n_epochs, int n_neg_samples, seed):
    # Convert to sparse coordinate matrix to efficiently access all 1-simplices
    weights = weights.tocoo()
    cdef:
        # For-loop over all sparse data
        Py_ssize_t n_sdata = len(weights.data), k
        int[:] rows = weights.row
        int[:] cols = weights.col
        double[:] prs = weights.data

        double alpha = 1.0  # learning rate
        int i, j, c  # sampled indices
        double square_dist, diff1, diff2, coeff

        mt19937 rng = mt19937(seed)  # random number generator
        uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0, 1.0)
        uniform_int_distribution[int] neg_dist = uniform_int_distribution[int](0, Y.shape[0] - 1)

    cdef double a, b
    a, b = fit_phi(min_dist, spread)

    # Precompute some values and manually inline gradient coefficients
    cdef double neg_two_ab = -2.0 * a * b, sd_pb, sd_pbm1, two_b = 2.0 * b, adj_sd

    for e in tqdm.trange(1, n_epochs + 1):
        for k in range(0, n_sdata, 1):
            # Sample this 1-simplex with probability p
            if dist(rng) <= prs[k]:
                i = rows[k]
                j = cols[k]
                diff1 = Y[i, 0] - Y[j, 0]
                diff2 = Y[i, 1] - Y[j, 1]
                square_dist = diff1 * diff1 + diff2 * diff2
                if square_dist != 0.0:
                    sd_pbm1 = square_dist ** (b-1)
                    sd_pb = square_dist ** b
                    coeff = neg_two_ab * sd_pbm1 / (1.0 + a * sd_pb)
                    Y[i, 0] += alpha * fast_clip(coeff * diff1)
                    Y[i, 1] += alpha * fast_clip(coeff * diff2)

                for _ in range(0, n_neg_samples, 1):
                    # Randomly sample from Y
                    c = neg_dist(rng)
                    if c != i:  # not self
                        diff1 = Y[i, 0] - Y[c, 0]
                        diff2 = Y[i, 1] - Y[c, 1]
                        square_dist = diff1 * diff1 + diff2 * diff2
                        if square_dist == 0.0:
                            Y[i, 0] += alpha * 4.0
                            Y[i, 1] += alpha * 4.0
                        else:
                            adj_sd = 1E-4 + square_dist
                            sd_pb = square_dist ** b
                            coeff = two_b / (adj_sd * (1.0 + a * sd_pb))
                            Y[i, 0] += alpha * fast_clip(coeff * diff1)
                            Y[i, 1] += alpha * fast_clip(coeff * diff2)

        alpha = 1.0 - float(e)/float(n_epochs)

    return Y

# Optimize embedding of general dimensions
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optimize_embedding(weights, double[:, :] Y, min_dist, spread, n_epochs, int n_neg_samples, seed):
    weights = weights.tocoo()
    cdef:
        Py_ssize_t n_sdata = len(weights.data), k
        int[:] rows = weights.row
        int[:] cols = weights.col
        double[:] prs = weights.data

        int dim = Y.shape[1], d

        int i, j, c
        double square_dist, coeff

        mt19937 rng = mt19937(seed)
        uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0, 1.0)
        uniform_int_distribution[int] neg_dist = uniform_int_distribution[int](0, Y.shape[0] - 1)

    cdef double alpha = 1.0

    cdef double a, b
    a, b = fit_phi(min_dist, spread)

    for e in tqdm.trange(1, n_epochs + 1):
        for k in range(0, n_sdata, 1):
            if dist(rng) <= prs[k]:
                i = rows[k]
                j = cols[k]
                square_dist = square_euclidean_dist(Y[i, :], Y[j, :])
                if square_dist != 0.0:
                    coeff = grad_coeff_phi(square_dist, a, b)
                    for d in range(0, dim, 1):
                        Y[i, d] += alpha * fast_clip(coeff * (Y[i, d] - Y[j, d]))

                for _ in range(0, n_neg_samples, 1):
                    # Randomly sample from Y
                    c = neg_dist(rng)
                    if c != i:
                        square_dist = square_euclidean_dist(Y[i, :], Y[c, :])
                        if square_dist == 0.0:
                            for d in range(0, dim, 1):
                                Y[i, d] += alpha * 4.0
                        else:
                            coeff = grad_coeff_neg_phi(square_dist, a, b)
                            for d in range(0, dim, 1):
                                Y[i, d] += alpha * fast_clip(coeff * (Y[i, d] - Y[c, d]))

        alpha = 1.0 - float(e)/float(n_epochs)

    return Y

# Fit phi by sampling psi (curve)
def fit_phi(min_dist, spread):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
