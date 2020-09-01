import numpy as np
import time


def init_centers(x, k):
    """
    Randomly samples k observations from X as centers.
    Returns these centers as a (k x d) numpy array.
    """
    samples = np.random.choice(len(x), size=k, replace=False)
    return x[samples, :]


def compute_d2(x, centers):
    m = len(x)
    k = len(centers)
    s = np.empty((m, k))
    if centers.ndim == 1:
        for i in range(m):
            s[i, :] = np.linalg.norm(x[i, :] - centers, ord=2) ** 2
    elif centers.ndim > 1:
        for i in range(m):
            s[i, :] = np.linalg.norm(x[i, :] - centers, ord=2, axis=1) ** 2
    else:
        raise Exception('{} is invalid:'.format(centers))
    return s


def compute_distances_no_loops(x, centers):
    # X: mxd
    # centers: kxd
    if centers.ndim == 1:
        centers = centers.reshape(len(centers), 1)
        dists = -2 * np.dot(x, centers) + np.sum(centers ** 2) + np.sum(x**2, axis=1)[:, np.newaxis]
    else:
        dists = -2 * np.dot(x, centers.T) + np.sum(centers ** 2, axis=1) + np.sum(x ** 2, axis=1)[:, np.newaxis]
    return dists


def assign_cluster_labels(s):
    return np.argmin(s, axis=1)


def update_centers(x, labels, k, old_centers, method='kmeans'):
    # X[:m, :d] == m points, each of dimension d
    # label[:m] == cluster labels
    m, d = x.shape
    assert m == len(labels)
    assert (min(labels) >= 0)

    if method == 'kmeans':
        centers = np.empty((k, d))
        for j in range(k):
            # Compute the new center of cluster j,
            # i.e., centers[j, :d].
            centers[j, :] = np.mean(x[labels == j, :], axis=0)
        return centers
    elif method == 'kmedoids' and old_centers.size > 0:
        centers = old_centers.copy()
        for j in range(k):
            clusters = x[labels == j, :].copy()
            s = compute_distances_no_loops(clusters, centers[j, :])
            s_all = compute_distances_no_loops(clusters, clusters)
            old_wcss = wcss(s)
            new_wcss = np.amin(np.sum(s_all, axis=1))
            if old_wcss > new_wcss:
                centers[j] = clusters[np.argmin(np.sum(s_all, axis=1)), :].copy()

        return centers
    else:
        raise Exception('Input is invalid')


def wcss(s):
    return np.sum(np.amin(s, axis=1))


def has_converged(old_centers, centers):
    return set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])


def clustering(x, k, starting_centers=None, max_steps=np.inf, method='kmeans'):
    if starting_centers is None:
        centers = init_centers(x, k)
    elif starting_centers.size > 0 and method == 'kmeans':
        centers = starting_centers
    else:
        raise Exception('Input {} is invalid'.format(method))

    converged = False
    labels = np.zeros(len(x))
    i = 1
    while (not converged) and (i <= max_steps):
        old_centers = centers.copy()
        s = compute_distances_no_loops(x, old_centers)  # calculate the distance between X and each center
        labels = assign_cluster_labels(s)  # assign the label based on min distance to each center
        centers = update_centers(x, labels, k, method=method, old_centers=old_centers)
        print("iteration", i, "WCSS = ", wcss(s))  # within-cluster-sum-of-squares
        if has_converged(old_centers, centers):
            converged = True
        i += 1
    return labels, centers


def mark_matches(a, b, exact=False):
    """
    Given two Numpy arrays of {0, 1} labels, returns a new boolean
    array indicating at which locations the input arrays have the
    same label (i.e., the corresponding entry is True).

    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as the same up to a swapping of the labels. This feature
    allows

      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]

    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    assert a.shape == b.shape
    a_int = a.astype(dtype=int)
    b_int = b.astype(dtype=int)
    assert ((a_int == 0) | (a_int == 1) | (a_int == 2)).all()
    assert ((b_int == 0) | (b_int == 1) | (b_int == 2)).all()

    exact_matches = (a_int == b_int)
    if exact:
        return exact_matches

    assert exact == False
    num_exact_matches = np.sum(exact_matches)
    if (2 * num_exact_matches) >= np.prod(a.shape):
        return exact_matches
    return exact_matches == False  # Invert


def count_matches(a, b, exact=False):
    """
    Given two sets of {0, 1} labels, returns the number of mismatches.

    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as similar up to a swapping of the labels. This feature
    allows

      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]

    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    matches = mark_matches(a, b, exact=exact)
    return np.sum(matches)


if __name__ == '__main__':

    B = np.random.random_sample(size=(100000, 2))
    k = 10
    method = 'kmedoids'
    print("size of the testing test: {}, k: {}, method: {}".format(B.size, k, method))

    # start_time = time.time()
    # _, _ = clustering(B, k, method='kmeans')
    # print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    _, _ = clustering(B, k, method=method)
    print("--- %s seconds ---" % (time.time() - start_time))

    # ### Test start ###
    # from sklearn import datasets
    # from sklearn.decomposition import PCA
    # iris = datasets.load_iris()
    # X_reduced = PCA(n_components=3).fit_transform(iris.data)
    # labels = iris.target
    # mylabels, centers = clustering(X_reduced, 3, method='kmedoids')
    # n_matches = count_matches(labels, mylabels)
    # print(n_matches,
    #       "matches out of",
    #       len(X_reduced), "data points",
    #       "(~ {:.1f}%)".format(100.0 * n_matches / len(labels)))
    # ### Test end ###

# ref1: https://github.com/briverse17/supernaive-kmedoids/blob/master/SuperNaive_kmedoids.ipynb
# ref2: https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
# ref3: https://www.youtube.com/watch?v=GApaAnGx3Fw
