import numpy as np
import time


def init_centers(x, k):
    """
    Randomly samples k observations from X as centers

    :return: centers as a (k x d) numpy array
    """
    samples = np.random.choice(len(x), size=k, replace=False)
    return x[samples, :]


def compute_d2(x, centers):
    """
    Compute the distance (l2 norm squared) between x (m x d) and centers (k x d)

    :return s: the distance matrix as a (m x k) numpy array
    """
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
    """
    An optimized approach to compute the distance (l2 norm squared) between x (m x d) and centers (k x d)

    :return dists: the distance matrix as a (m x k) numpy array
    """
    if centers.ndim == 1:
        centers = centers.reshape(len(centers), 1)
        dists = -2 * np.dot(x, centers) + np.sum(centers ** 2) + np.sum(x**2, axis=1)[:, np.newaxis]
    else:
        dists = -2 * np.dot(x, centers.T) + np.sum(centers ** 2, axis=1) + np.sum(x ** 2, axis=1)[:, np.newaxis]
    return dists


def assign_cluster_labels(s):
    """
    Create cluster labels for cluster assignments based on distance matrix S
    """
    return np.argmin(s, axis=1)


def update_centers(x, labels, k, old_centers, method='kmeans'):
    """
    Search for the new centers for each cluster to minimize the within cluster distance

    :param X: m points, each of dimension d
    :param labels: m cluster labels
    :return the new centers for each cluster as a (k x d) array
    """

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
    """
    Compute within cluster sum of squares of distance
    """
    return np.sum(np.amin(s, axis=1))


def has_converged(old_centers, centers):
    """
    Determine whether the algorithm is converged
    by checking if the old_centers and centers have become identical
    """
    return set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])


def clustering(x, k, starting_centers=None, max_steps=np.inf, method='kmeans'):
    """
    Implement the clustering algorithm, consisting of:
    (1) assigning each data point to a cluster
    (2) updating cluster center in each cluster
    """
    if starting_centers is None:
        centers = init_centers(x, k)
    elif starting_centers.size > 0 and method == 'kmeans':
        centers = starting_centers
    else:
        raise Exception('Input {} is invalid'.format(method))

    converged = False
    labels = np.zeros(len(x))
    i = 1
    log = []
    while (not converged) and (i <= max_steps):
        old_centers = centers.copy()
        s = compute_distances_no_loops(x, old_centers)  # calculate the distance between X and each center
        labels = assign_cluster_labels(s)  # assign the label based on min distance to each center
        centers = update_centers(x, labels, k, method=method, old_centers=old_centers)
        wcss_i = wcss(s)
        print("iteration", i, "WCSS = ", wcss_i)  # within-cluster-sum-of-squares
        log.append((i, wcss_i))
        if has_converged(old_centers, centers):
            converged = True
        i += 1
    return labels, centers, log


if __name__ == '__main__':
    # create a test dataset
    B = np.random.random_sample(size=(10000, 2))

    # set up the parameters for the run
    k = 10
    method = 'kmedoids'
    print("size of the testing test: {}, k: {}, method: {}".format(B.size, k, method))

    # kick off the clustering
    start_time = time.time()
    _, _, iteration_log = clustering(B, k, method=method)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(iteration_log)
