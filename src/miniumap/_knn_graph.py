
try:
    import pynndescent
    HAVE_PYNNDESCENT = True
except ImportError:
    HAVE_PYNNDESCENT = False

def approx_nearest_neighbors(X, k, metric='euclidean', random_seed=None):
    if not HAVE_PYNNDESCENT:
        raise NotImplementedError("kNN graph construction requires pynndescent package")

    index = pynndescent.NNDescent(X, metric=metric, n_neighbors=k, random_state=random_seed)
    knn_indices, knn_dists = index.neighbor_graph
    return knn_indices, knn_dists
