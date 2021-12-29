
from miniumap._knn_graph import approx_nearest_neighbors
from miniumap._embedding import (
    spectral_embedding,
    optimize_embedding,
    optimize_embedding2
)

import numpy as np
import scipy.sparse
import tqdm

import random

def umap(X, k=15, d=2, min_dist=0.1, spread=1.0, n_epochs=200, random_seed=None, n_neg_samples=5, metric='euclidean'):
    n_datapoints = X.shape[0]

    # Adjacency matrix of directed weighted kNN graph
    #     A[i, j] = Pr{ j belongs to the neighborhood of i }
    asymm_weights = scipy.sparse.lil_matrix((n_datapoints, n_datapoints), dtype=np.double)
    # Note that argknn[0] == arg(x, X) and knn_dists[0] == 0
    print("Computing nearest neighbor graph...")
    all_argknn, all_knn_dists = approx_nearest_neighbors(X, k, metric=metric, random_seed=random_seed)
    print("Finished.")

    print("Computing weights...")
    # For each row (datapoint) i in X, updates weights for neighbors of i (W[i, i[...]])
    for i in tqdm.trange(0, len(all_knn_dists)):
        rho = all_knn_dists[i][1]
        sigma = smooth_knn_dist(all_knn_dists[i], k, rho)
        for argxn, xn_dist in zip(all_argknn[i], all_knn_dists[i]):
            # Thus the maximum weight is 1 (along the diagonal and closest neighbors)
            asymm_weights[i, argxn] = np.exp(-max(0, xn_dist - rho) / sigma)
    del all_argknn, all_knn_dists
    print("Finished.")

    print("Computing spectral embedding...")
    # Symmetrize into undirected weighted kNN graph using probabilistic t-conorm
    asymm_weights = asymm_weights.tocsc()
    awt = asymm_weights.transpose()

    # A side note: the behavior of operator * in NumPy depends on the data type, so we
    # always refer to the explicit methods (multiply, dot and matmul)
    symm_weights = asymm_weights + awt - asymm_weights.multiply(awt)
    del asymm_weights, awt

    # Generate an initial low-dimensional representation
    init_repr = spectral_embedding(symm_weights, d)
    print("Finished.")

    print("Optimizing final representation...")
    # Optimize the final representation
    if random_seed is None:
        random_seed = random.getrandbits(32)
    if d == 2:
        final_repr = optimize_embedding2(symm_weights, init_repr, min_dist, spread, n_epochs, n_neg_samples, random_seed)
    else:
        final_repr = optimize_embedding(symm_weights, init_repr, min_dist, spread, n_epochs, n_neg_samples, random_seed)
    print("Finished.")

    return final_repr

def _lhs(knn_dists, sigma, rho):
    return np.sum(np.exp(-(knn_dists[1:] - rho) / sigma))

def smooth_knn_dist(knn_dists, k, rho):
    RHS = np.log2(k)
    EPS = 1E-4

    sigma_hi = 1.0
    while _lhs(knn_dists, sigma_hi, rho) < RHS:
        sigma_hi *= 2

    sigma_lo = 0.0
    while (sigma_hi - sigma_lo) > EPS:
        sigma_mid = (sigma_hi + sigma_lo) / 2
        if _lhs(knn_dists, sigma_mid, rho) > RHS:
            sigma_hi = sigma_mid
        else:
            sigma_lo = sigma_mid

    return sigma_mid
